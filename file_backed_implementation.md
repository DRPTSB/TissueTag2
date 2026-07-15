# File-Backed Low-RAM Mode: Implementation Summary

Branch: `file_backed_mode` (3 code commits on top of `main`: `0861d5e`,
`76a9956`, `b8d2876`; a separate `8775100` adds this file, the original
plan, and a full chat export).

This document summarizes the implementation of an opt-in, file-backed (Zarr +
Dask + Xarray) low-RAM mode for TissueTag2, following `file_backed_plan.md`.
It also records where the implementation deliberately departs from that plan,
and a couple of real bugs found and fixed along the way.

## Scope and design decisions

Agreed up front, before writing any code:

- **Opt-in, additive.** The existing in-memory numpy pipeline is completely
  unchanged by default. Nothing in `tissue_tag.file_backed` is imported
  unless a caller explicitly asks for file-backed mode.
- **The pixel classifier (`pixel_label_classifier`) stays in-memory**, with a
  documented, warned fallback. `skimage.feature.multiscale_basic_features`
  (Gaussian derivatives up to sigma=16) is not chunk/dask-aware; making it so
  would be a large, separate project. This was the one function that was
  *not* later upgraded to be chunk-aware (see "What stayed in-memory" below).
- **Testing:** no real 50,000x50,000 test image was available, so validation
  uses a synthetic image generated directly to an on-disk Zarr store, and
  drives the real Panel/Bokeh widget wiring programmatically (script-level,
  not a browser).

## Architecture

- **`image`** (the morphology image) is read-mostly, so it's represented as
  a lazy, dask-backed `xarray.DataArray` (dims `y, x, band`) reading directly
  from an on-disk Zarr store. HoloViews/Datashader (`regrid`) consume it
  directly and only materialise pixels needed for the current viewport/zoom.
- **`label_image`** needs random-access in-place writes whenever a user
  commits a drawn stroke, which dask arrays don't support (read-only/lazy).
  It's kept as a writable `zarr.Array` opened in `r+` mode
  (`file_backed.WritableLabelStore`), with a throwaway dask/xarray *view* of
  the same store built on demand for rendering. Because the view always
  reads from the live store, it reflects the latest on-disk writes without
  ever holding a full-resolution copy in memory.
- **Coordinate convention:** Zarr-backed `xarray.DataArray`s use
  `y`/`x` coordinates offset by `+0.5` (cell centres) so HoloViews infers
  exactly the same `(0, 0, w, h)` pixel-edge bounds as the numpy path's
  `bounds=` argument — verified so drawn-stroke pixel coordinates map
  identically between the two code paths.
- **Bounding-box-scoped writes**, not whole-array copies. Every write
  (interactive stroke, background-intensity spot, gene-marker cell) reads
  only the bounding box it touches, modifies it, and writes it back. Undo
  works by replaying the pre-write contents of each touched box in reverse
  order — no full-array copy is ever taken for revert.

## Files changed

### `tissue_tag/file_backed.py` (new, 346 lines)

Core Zarr/Dask/Xarray primitives:

| Function/class | Purpose |
|---|---|
| `array_to_zarr` / `zeros_zarr` | Persist a numpy array (or create an all-zero array) to an on-disk Zarr store, row-band by row-band; never holds a second full copy in RAM. |
| `image_dataarray` / `label_dataarray` | Lazy, dask-backed `xarray.DataArray` views for reading, with the `+0.5` coordinate offset described above. |
| `WritableLabelStore` | Wraps a writable `zarr.Array`. `read_block`/`write_block` for bounding-box I/O; `write_masked(y0,y1,x0,x1,local_mask,value,preserve_existing=)` is the shared primitive behind every bbox-scoped write in the codebase (drawn strokes, background spots, gene markers) — `preserve_existing=True` only overwrites currently-zero pixels, used wherever a lower-priority label must not clobber an existing one. |
| `tiff_to_zarr_store` | Genuinely zero-copy lazy TIFF/OME-TIFF ingestion via `tifffile`'s `aszarr=True`, for users with pyramidal source data. |
| `configure_dask_for_low_ram` | Caps dask's threaded-scheduler worker count (default 4). Necessary because regrid rendering fans out into many small per-chunk tasks, and dask's default scheduler runs up to `os.cpu_count()` of them concurrently — on the 120-core test box, that alone was enough to blow a multi-GB budget even with small per-task chunks. |

### `tissue_tag/io.py`

`TissueTagAnnotation` (dataclass) gains:
- `image_store` / `label_store: Optional[str]` fields and a `file_backed`
  property.
- `to_file_backed(file_backed_dir, chunks=None, overwrite=True)`: one-time
  conversion of `image`/`label_image` to on-disk Zarr stores + lazy views;
  safe to call on an already-partially-file-backed object.
- `label_writer()`: returns a `WritableLabelStore` bound to the object's
  label store.
- `refresh_label_view()`: rebuilds the lazy `label_image` view after writes
  made through `label_writer()`.

`save_annotation`/`load_annotation`: when file-backed, persist just the
Zarr store *paths* into the HDF5 file instead of re-serialising the arrays
(verified: 8,248 bytes instead of hundreds of MB). `load_annotation` reopens
them as lazy views.

**Loaders (`read_image`, `read_visium`, `read_visium_hd`, `read_xenium`)**
all gained `file_backed=False, file_backed_dir=None` parameters, so a large
image can be loaded straight into file-backed mode in one call instead of
loading in-memory and calling `to_file_backed()` as a separate step:

```python
tta = tt.read_image(path, ppm_image=1, file_backed=True)
```

Each loader still decodes/resizes/blends the image fully in memory first —
PIL/tifffile have no chunked reader for PNG/JPEG, and contrast enhancement,
vH&E blending, and Xenium channel stacking all need the whole array anyway
— but when `file_backed=True` a new shared helper, `io._finalize_annotation`,
persists the result to an on-disk Zarr store and the in-memory array is
dropped immediately afterward, before the caller ever holds it alongside a
second full-resolution copy later on (annotator/segmenter, classifier).
`file_backed_dir` is optional; if not given, `_finalize_annotation` creates
a fresh temporary directory (`tempfile.mkdtemp(prefix="tissue_tag_")`).

The directory-for-on-disk-stores parameter is named `file_backed_dir`
consistently across the whole public API (`TissueTagAnnotation.to_file_backed`,
`annotator()`, `segmenter()`, and all four loaders) — it was originally
called `work_dir` and was renamed for API consistency partway through
implementation.

Verified against real sample data for `read_image` and `read_visium`
(file-backed output pixel-identical to in-memory, both with an
auto-generated temp directory and an explicit `file_backed_dir`).
`read_visium_hd` and `read_xenium` got the identical code pattern but
weren't separately exercised against real data -- no local sample dataset
for either was available in this environment.

### `tissue_tag/annotation.py` (largest set of changes)

- **`base_image_element` / `label_image_overlay` / `label_image_element`**
  accept dask-backed `xarray.DataArray` directly (detected via
  `_is_xarray_backed`), feeding `hv.RGB`/`hv.Image` and datashader's
  `regrid`. The label-masking step (`mask_hidden_labels`, used to toggle
  annotation visibility) applies its LUT via `dask.array.map_blocks` instead
  of plain numpy fancy indexing, so it stays chunk-aware too.
- **`annotator()` / `segmenter()`** gained `file_backed`/`file_backed_dir`
  parameters. When enabled, drawn-stroke commits go through
  `_write_polygon_strokes_file_backed` / `_revert_polygon_strokes_file_backed`,
  which compute each stroke's bounding box and read/write only that region
  — including correct bbox-scoped undo (reverted in reverse write order).
- **`_ensure_in_memory`**: materialisation shim for functions that remain
  numpy-only. Converts a file-backed input to plain numpy on entry with a
  `UserWarning`, used by `pixel_label_classifier` and `median_filter`.
  Read-only plotting helpers (`rgb_from_labels`, `plot_labels`) instead
  materialise a *local* copy so they never silently strip a caller's
  file-backed state.
- **`gene_labels_from_adata`** (see "Round 2" below) has a parallel
  file-backed branch using two new helpers:
  - `_write_disks_batched_file_backed(writer, points, r, value, preserve_existing)`:
    writes many same-radius disks, batched by on-disk chunk (see "Bug #2"
    below for why batching matters).
  - `_background_labels_intensity_file_backed(writer, image, r, ...)`:
    file-backed counterpart of `background_labels_intensity`, using a single
    `dask.array.vindex` gather to test all sparse grid points' brightness in
    one batched, chunk-aware call.

### `tissue_tag/organaxis.py`

`get_annotations_for_objects`: for a file-backed `label_image`, looks up
label values via `label_image.data.vindex[rows, cols]` (dask's paired/
vectorized indexing) instead of `arr[rows, cols]`. This isn't just a memory
optimisation — plain numpy-style fancy indexing on an xarray-backed array
performs *outer-product* indexing, not paired indexing, so the naive
translation would have silently produced wrong results, not just been slow.

### `setup.py`

New `file_backed` extra: `xarray`, `dask[array]`, `zarr>=3`, `dask-ml`.
Install with `pip install -e .[file_backed]`.

### `tests/validate_file_backed_mode.py` (new, 338 lines)

Standalone, rerunnable script (no pytest dependency) that:
1. Writes a 35,000x35,000 synthetic RGBA image directly to Zarr, row-band by
   row-band (never holds the full array in RAM even to generate it).
2. Builds and renders the real `annotator()` app in file-backed mode.
3. Drives the actual `CustomFreehandDraw` streams and clicks the real
   Update/Revert buttons to commit and undo a stroke, checking only the
   touched region changed.
4. Round-trips `save_annotation`/`load_annotation`.
5. Exercises the classifier fallback on a small crop, checking it warns and
   doesn't mutate the file-backed original.
6. Exercises `gene_labels_from_adata` + `assign_annotation_label_to_positions`
   at full scale.

Tracks peak RSS throughout via a background sampler thread and fails loudly
(non-zero exit) on any correctness or memory-budget violation. Deliberately
writes its synthetic data under `/var/tmp` rather than `/tmp`, having
discovered the test box's `/tmp` is `tmpfs` (RAM-backed) — writing "to disk"
there would have silently inflated the RAM measurement.

## Bugs found and fixed along the way

These surfaced only because the feature was validated at real scale rather
than just unit-tested; none were assumptions from the plan.

**Bug #1 — `regrid` silently materialises the whole image.** HoloViews'
`regrid` operation never passes datashader a `max_mem` budget. Without one,
`datashader.Canvas.raster()`'s automatic chunking (`compute_chunksize`) falls
back to using the source array's *own* dask chunksize verbatim as the
*output*-space chunk grid. Since our on-disk chunks (~2048px) are far larger
than a typical downsampled overview (a few hundred px), that fallback
collapses the "chunked" resample into a single task spanning the entire
source array — an 11-12GB memory spike was observed on the first real
35,000x35,000 render. Fixed by patching `datashader.Canvas.raster`
(`annotation._low_ram_canvas_raster`) to default `max_mem=64MB` whenever a
caller doesn't specify one; harmless for plain-numpy sources. Documented in
detail in the code since it's a non-obvious interaction between two
third-party libraries.

**Bug #2 — unbounded thread growth from one-write-per-point.** The first
version of the gene-label background-spot writer did one `read_block`/
`write_block` round trip per sparse grid point (tens of thousands for a
large image). At that call volume, zarr's synchronous/async bridge span up
an *unbounded* number of OS threads rather than merely running slowly —
confirmed live via `/proc/<pid>/task/` showing 260+ threads and climbing,
CPU time barely advancing. Fixed by `_write_disks_batched_file_backed`,
which groups points by which on-disk chunk they fall in (padded by the disk
radius to catch spillover) and does one read+write per *touched chunk*
instead of per point — cutting round trips from ~40,000 to ~100 for the
35,000x35,000 test case, and turning a hang into a 44.6s run.

**Bug #3 — `gene_labels_from_adata` broken on pandas >=2.2 (pre-existing,
unrelated to file-backed mode).** `groupby("expression").apply(...)` now
drops the grouping column from its result by default, so the following
`sort_values("expression")` raised `KeyError` unconditionally — this
function could not run at all with a modern pandas, in any mode. Found
because it blocked testing the file-backed path. Fixed by replacing the
groupby-then-shuffle-then-sort with an equivalent shuffle-then-stable-sort,
which doesn't rely on the now-removed behaviour.

## What stayed in-memory (by design)

Only `pixel_label_classifier`: `skimage.feature.multiscale_basic_features`
is not chunk-aware, and making it so (via `dask.array.map_overlap` with
padding for the Gaussian derivative windows) was scoped out at the start as
a materially larger, separate effort. It still respects file-backed mode's
`_ensure_in_memory` shim (materialises a copy, warns, never mutates a
`copy=True` caller's original), and existing `downsampling_factor` support
is the recommended way to manage its memory cost on very large images.

`median_filter`, `rgb_from_labels` also fall back the same way (thin
numpy-only wrappers around `skimage`/matplotlib) — but `gene_labels_from_adata`
and `assign_annotation_label_to_positions`, initially placed in this same
"materialise on entry" bucket, were later upgraded to be fully chunk-aware
(see above) since both are fundamentally sparse-point operations bounded by
cell/spot count rather than image size.

## Validation results (35,000 x 35,000 synthetic image)

Naive full in-memory RGBA + label load for this image: ~5,841 MB.

| Step | Peak RSS | Notes |
|---|---|---|
| 1. Generate synthetic Zarr stores | 1,151 MB | Baseline process + import overhead dominates. |
| 2. Build + render annotator (datashader regrid) | 1,802 MB | 313.6s — see Bug #1 fix; still the slowest step. |
| 3. Draw / Update / Revert via real UI wiring | 1,876 MB | Correctness verified: touched region only, revert exact. |
| 4. `save_annotation`/`load_annotation` round trip | 1,876 MB | HDF5 file: 8,248 bytes. |
| **Steps 1-4 combined** | **1,876 MB** (budget: 3,072 MB) | vs. ~5,841 MB naive. |
| 5. `pixel_label_classifier` fallback (800x800 crop) | n/a (separate budget) | Warns, materialises, doesn't mutate original. |
| 6. `gene_labels_from_adata` + `assign_annotation_label_to_positions` | 2,012 MB | 44.6s; exact match vs. in-memory reference across override/preserve-existing combinations. |

Regression-tested: the original, non-file-backed `annotator()`/`segmenter()`
path was re-run after every round of changes and produces identical results
to before this branch existed.

## Usage

Simplest: load straight into file-backed mode.

```python
import tissue_tag as tt
from tissue_tag.annotation import annotator

# file_backed_dir is optional -- omit it to use a fresh temporary directory
tta = tt.read_image("huge_image.tif", ppm_image=1, plot=False,
                    file_backed=True, file_backed_dir="/scratch/my_annotation")
tta.annotation_map = {"cortex": "green", "medulla": "blue"}

app = annotator(tta, use_datashader=True)  # file_backed inferred from tta.file_backed
```

Or load in-memory as usual and opt in later, either explicitly:

```python
tta = tt.read_image("huge_image.tif", ppm_image=1, plot=False)
tta.to_file_backed("/scratch/my_annotation")
app = annotator(tta, use_datashader=True)
```

...or by letting `annotator()`/`segmenter()` do the conversion on first use:

```python
tta = tt.read_image("huge_image.tif", ppm_image=1, plot=False)
app = annotator(tta, use_datashader=True, file_backed=True, file_backed_dir="/scratch/my_annotation")
```

`read_visium`, `read_visium_hd`, and `read_xenium` accept the same
`file_backed`/`file_backed_dir` pair.

Install the extra: `pip install -e .[file_backed]`.

Run the validation script: `python tests/validate_file_backed_mode.py
[--side 35000] [--threshold-mb 3072]` (needs `psutil`, not a package
dependency).
