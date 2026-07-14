"""
Validation script for TissueTag2's file-backed (Zarr + Dask + Xarray) low-RAM mode.

Not a pytest suite (the repo has none yet) -- a standalone, rerunnable script that
exercises the real annotator()/segmenter() UI wiring against a synthetic large image
while tracking this process's peak resident memory, and fails loudly (non-zero exit)
on any correctness or memory-budget violation.

Requires the 'file_backed' extra:  pip install -e .[file_backed]
Also needs 'psutil' (not a package dependency; only used by this script).

Usage:
    python tests/validate_file_backed_mode.py [--side 35000] [--threshold-mb 3072]

By default this writes its synthetic Zarr stores under a temp directory on a real
(non-tmpfs) disk -- see `_pick_work_root()` -- since a tmpfs-backed /tmp would make
"on disk" writes actually consume RAM, defeating the point of the measurement.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
from collections import OrderedDict

import numpy as np

try:
    import psutil
except ImportError:
    sys.exit("This script needs psutil to measure memory: pip install psutil")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import holoviews as hv
from holoviews.operation import datashader as hd

hv.extension('bokeh')

import tissue_tag as tt
from tissue_tag import file_backed as fb
from tissue_tag.annotation import annotator, pixel_label_classifier

CHUNKS = (2048, 2048)

FAILURES = []


def check(condition, message):
    if condition:
        print(f"  OK: {message}")
    else:
        print(f"  FAIL: {message}")
        FAILURES.append(message)


class PeakRSSMonitor:
    """Background sampler tracking this process's peak resident set size (RSS)."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            rss_mb = self.process.memory_info().rss / (1024 ** 2)
            self.peak_mb = max(self.peak_mb, rss_mb)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_mb = self.process.memory_info().rss / (1024 ** 2)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _is_tmpfs(path, mount_out):
    """Whether `path` sits on a filesystem `mount` reports as tmpfs, walking up to
    the nearest mount point mentioned in `mount_out`."""

    path = os.path.abspath(path)
    mounts = {}
    for line in mount_out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1] == "on" and parts[3] == "type":
            mounts[parts[2]] = parts[4]

    while path != os.path.dirname(path):
        if path in mounts:
            return mounts[path] == "tmpfs"
        path = os.path.dirname(path)
    return mounts.get("/") == "tmpfs"


def _pick_work_root():
    """Prefer a real, non-tmpfs-backed disk for the synthetic Zarr stores, so that
    writing "to disk" doesn't just push data into RAM via a tmpfs /tmp."""

    try:
        import subprocess
        mount_out = subprocess.run(["mount"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        mount_out = ""

    for candidate in (tempfile.gettempdir(), "/var/tmp", "."):
        path = os.path.abspath(candidate)
        if not _is_tmpfs(path, mount_out):
            return path
    return os.path.abspath(".")


def generate_large_zarr(work_dir, side, chunks=CHUNKS):
    """
    Write a synthetic RGBA image directly to an on-disk Zarr store, one row-band
    (chunk row) at a time, plus an all-zero label store -- so this generation step
    itself never holds more than a single band in memory, regardless of `side`.
    """

    import zarr

    image_path = os.path.join(work_dir, "image.zarr")
    label_path = os.path.join(work_dir, "label.zarr")

    z_img = zarr.create_array(
        store=image_path, shape=(side, side, 4), dtype=np.uint8,
        chunks=chunks + (4,), overwrite=True,
    )
    step = chunks[0]
    for y0 in range(0, side, step):
        y1 = min(y0 + step, side)
        band_h = y1 - y0
        y_idx = (np.arange(y0, y1, dtype=np.uint16) % 256).astype(np.uint8)[:, None]
        x_idx = (np.arange(side, dtype=np.uint16) % 256).astype(np.uint8)[None, :]
        band = np.empty((band_h, side, 4), dtype=np.uint8)
        band[:, :, 0] = y_idx
        band[:, :, 1] = x_idx
        band[:, :, 2] = ((y_idx.astype(np.uint16) + x_idx.astype(np.uint16)) % 256).astype(np.uint8)
        band[:, :, 3] = 255
        z_img[y0:y1] = band
        del band

    fb.zeros_zarr((side, side), label_path, dtype=np.uint8, chunks=chunks, overwrite=True)

    return image_path, label_path


def run(work_dir, side, threshold_mb):
    print(f"Working directory: {work_dir}")
    naive_mb = (side ** 2 * 5) / 1024 ** 2  # RGBA (4 bytes) + label (1 byte) per pixel
    print(f"Synthetic image size: {side} x {side}  "
          f"(naive full in-memory RGBA+label would need ~{naive_mb:.0f} MB)")

    monitor = PeakRSSMonitor()
    with monitor:
        # --- Step 1: generate on disk, chunk by chunk ---
        t0 = time.time()
        image_path, label_path = generate_large_zarr(work_dir, side)
        print(f"\nStep 1 -- generate synthetic Zarr stores ({time.time() - t0:.1f}s); "
              f"peak RSS so far: {monitor.peak_mb:.0f} MB")

        tta = tt.TissueTagAnnotation(
            image=None, ppm=1.0,
            annotation_map=OrderedDict({"cortex": "green", "medulla": "blue"}),
        )
        tta.image_store = image_path
        tta.label_store = label_path
        tta.image = fb.image_dataarray(image_path)
        tta.label_image = fb.label_dataarray(label_path)
        check(tta.file_backed, "TissueTagAnnotation reports file_backed=True")

        # --- Step 2: build + render annotator elements (datashader regrid) ---
        t0 = time.time()
        app = annotator(tta, plot_size=512, use_datashader=True, file_backed=True, work_dir=work_dir)
        row = app[0]
        update_button, revert_button = row[1], row[2]
        hv.render(app[2].object)
        print(f"\nStep 2 -- build + render annotator ({time.time() - t0:.1f}s); "
              f"peak RSS so far: {monitor.peak_mb:.0f} MB")

        # --- Step 3: simulated draw -> Update -> Revert through the real UI wiring ---
        streams_by_tooltip = {}
        for _, streams in hv.streams.Stream.registry.items():
            for s in streams:
                if type(s).__name__ == 'CustomFreehandDraw':
                    streams_by_tooltip[s.tooltip] = s
        for s in streams_by_tooltip.values():
            s.event(data={'xs': [], 'ys': []})

        mid = side // 2
        sy0, sy1, sx0, sx1 = mid, mid + 100, mid, mid + 100
        streams_by_tooltip['cortex'].event(data={
            'xs': [[sx0, sx1, sx1, sx0]], 'ys': [[sy0, sy0, sy1, sy1]],
        })
        update_button.param.trigger('value')

        region = tta.label_image[sy0 + 5:sy1 - 5, sx0 + 5:sx1 - 5].compute().values
        check(set(np.unique(region)) == {2}, "drawn stroke committed with the expected label value")
        far = tta.label_image[0:10, 0:10].compute().values
        check(np.all(far == 0), "region far from the stroke is untouched")

        revert_button.param.trigger('value')
        reverted = tta.label_image[sy0 + 5:sy1 - 5, sx0 + 5:sx1 - 5].compute().values
        check(np.all(reverted == 0), "revert restores the pre-update state")
        print(f"\nStep 3 -- draw / update / revert; peak RSS so far: {monitor.peak_mb:.0f} MB")

        # re-draw for the save/load check below
        streams_by_tooltip['cortex'].event(data={
            'xs': [[sx0, sx1, sx1, sx0]], 'ys': [[sy0, sy0, sy1, sy1]],
        })
        update_button.param.trigger('value')

        # --- Step 4: save_annotation() / load_annotation() round trip ---
        t0 = time.time()
        h5_path = os.path.join(work_dir, "annotation.h5")
        tta.save_annotation(h5_path)
        h5_size = os.path.getsize(h5_path)
        check(h5_size < 10 * 1024 * 1024,
              f"saved HDF5 metadata file stays small ({h5_size} bytes -- arrays not re-serialised)")

        loaded = tt.load_annotation(h5_path)
        check(loaded.file_backed, "loaded annotation is file-backed")
        loaded_region = loaded.label_image[sy0 + 5:sy1 - 5, sx0 + 5:sx1 - 5].compute().values
        check(set(np.unique(loaded_region)) == {2}, "loaded annotation reflects the committed stroke")
        print(f"\nStep 4 -- save_annotation/load_annotation round trip ({time.time() - t0:.1f}s); "
              f"peak RSS so far: {monitor.peak_mb:.0f} MB")

    print(f"\nPeak RSS across steps 1-4: {monitor.peak_mb:.0f} MB "
          f"(threshold {threshold_mb} MB; naive full in-memory load would need ~{naive_mb:.0f} MB)")
    check(monitor.peak_mb < threshold_mb, f"peak RSS during file-backed steps stayed under {threshold_mb} MB")

    # --- Step 5: documented classifier fallback on a *small crop* -- deliberately
    # out of the RSS budget above; this step is scoped to materialise in RAM (see
    # annotation._ensure_in_memory and pixel_label_classifier's docstring). ---
    print("\nStep 5 -- pixel_label_classifier fallback on a small crop (expected to materialise)")
    crop_side = 800
    crop_image = tta.image[0:crop_side, 0:crop_side].compute().values
    crop_tta = tt.TissueTagAnnotation(
        image=crop_image, ppm=1.0,
        label_image=np.zeros((crop_side, crop_side), dtype=np.uint8),
        annotation_map=OrderedDict({"cortex": "green", "medulla": "blue"}),
    )
    crop_tta.label_image[50:150, 50:150] = 1
    crop_tta.label_image[400:500, 400:500] = 2

    crop_work_dir = tempfile.mkdtemp(prefix="tt_filebacked_crop_", dir=os.path.dirname(work_dir))
    try:
        crop_tta.to_file_backed(crop_work_dir)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pixel_label_classifier(crop_tta, downsampling_factor=2, plot=False, copy=True)
            check(
                any("not chunk-aware" in str(warning.message) for warning in caught),
                "classifier fallback warns about materialising to RAM",
            )
        check(not out.file_backed, "classifier output is plain in-memory (as documented)")
        check(crop_tta.file_backed, "original crop annotation is untouched (copy=True)")
        check(out.label_image.shape == (crop_side, crop_side), "classifier output has the expected shape")
    finally:
        shutil.rmtree(crop_work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=35000,
                        help="Side length (pixels) of the synthetic square test image.")
    parser.add_argument("--threshold-mb", type=int, default=3072,
                        help="Fail if peak RSS during steps 1-4 exceeds this many MB.")
    args = parser.parse_args()

    work_root = _pick_work_root()
    work_dir = tempfile.mkdtemp(prefix="tt_filebacked_validate_", dir=work_root)
    print(f"Using {work_root} for synthetic Zarr stores (avoiding a tmpfs-backed /tmp).")
    try:
        run(work_dir, args.side, args.threshold_mb)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for message in FAILURES:
            print(f"  - {message}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
