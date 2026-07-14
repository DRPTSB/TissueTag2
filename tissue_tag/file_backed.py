"""
Low-RAM, file-backed (Zarr + Dask + Xarray) storage helpers for TissueTag2.

This module is opt-in: nothing in the default in-memory (numpy) pipeline
imports it, and it is only pulled in when a caller explicitly asks for
file-backed mode (e.g. ``annotator(..., file_backed=True)`` or
``TissueTagAnnotation.to_file_backed(...)``).

Design summary
--------------
* The morphology ``image`` is read-mostly, so it is represented as a
  dask-backed ``xarray.DataArray`` (dims ``y, x, band``) reading straight
  from an on-disk Zarr store. Holoviews/Datashader (``regrid``) consume this
  directly and only materialise the pixels needed for the current viewport
  and zoom level.
* The ``label_image`` needs random-access in-place writes whenever the user
  commits a drawn stroke, which dask arrays do not support (they are
  read-only/lazy). It is therefore kept as a writable ``zarr.Array`` opened
  in ``r+`` mode (:class:`WritableLabelStore`), with a throwaway
  dask/xarray *view* of that same store created on demand for rendering.
  Because the view always reads from the live store, it reflects the latest
  on-disk writes without ever holding a full-resolution copy in memory.

Requires the ``file_backed`` extra (``pip install tissue-tag[file_backed]``):
xarray, dask[array], zarr>=3, dask-ml.
"""

import os

import numpy as np

try:
    import dask
    import zarr
    import dask.array as da
    import xarray as xr
except ImportError as e:  # pragma: no cover - exercised only when extra is missing
    raise ImportError(
        "File-backed mode requires the 'file_backed' extra. "
        "Install it with `pip install tissue-tag[file_backed]`."
    ) from e

# Default on-disk chunk size (pixels) along the y/x axes. 2048x2048 keeps a
# single chunk's worth of uint8 RGBA data (~16MB) comfortably small while
# still being large enough that datashader/regrid don't need to touch an
# unreasonable number of chunks for a full-extent overview.
DEFAULT_CHUNKS = (2048, 2048)

BAND_NAMES = ['R', 'G', 'B', 'A']

# Default cap on dask's threaded-scheduler worker count for this process, applied by
# configure_dask_for_low_ram() (called automatically from TissueTagAnnotation.to_file_backed()).
#
# Why this matters: rendering a regrid'd view fans out into many small per-chunk tasks (see
# annotation.REGRID_CHUNK_SIZE), each holding one input slice in memory for the duration of that
# task. Dask's default threaded scheduler runs up to `os.cpu_count()` of those tasks concurrently,
# so peak memory scales with core count -- on a many-core machine (seen: 120 cores) that alone was
# enough to blow well past a multi-GB budget even with small per-task chunks. Capping the worker
# count bounds peak memory to (worker_count * per-task size) regardless of host core count, at the
# cost of some rendering throughput -- an appropriate trade-off for a mode whose whole point is
# staying under a fixed RAM budget.
DEFAULT_MAX_DASK_WORKERS = 4


def configure_dask_for_low_ram(max_workers=DEFAULT_MAX_DASK_WORKERS):
    """
    Cap dask's (process-global) threaded-scheduler worker count, so that rendering a file-backed
    image/label overlay can't fan out into a number of concurrent, memory-holding tasks that scales
    with the host's core count. See DEFAULT_MAX_DASK_WORKERS for why. Safe to call repeatedly; pass
    `max_workers=None` to leave dask's scheduler configuration untouched (e.g. if the caller wants
    to manage this themselves, perhaps because dask is also used elsewhere in the same process).
    """

    if max_workers is not None:
        dask.config.set(scheduler='threads', num_workers=max_workers)


def _chunk_shape(shape, chunks):
    """Extend a 2D (y, x) chunk spec with any trailing (e.g. band) axes, which
    are always kept unchunked (a single RGBA pixel's bands are never split
    across chunks)."""

    chunks = tuple(chunks)
    if len(shape) > len(chunks):
        chunks = chunks + tuple(shape[len(chunks):])
    return tuple(min(c, s) for c, s in zip(chunks, shape))


def array_to_zarr(array, zarr_path, chunks=DEFAULT_CHUNKS, overwrite=True):
    """
    Persist an in-memory numpy array to an on-disk Zarr store, row-chunk by
    row-chunk, so this conversion itself never needs a second full-size copy
    beyond the source array (which the caller is free to ``del`` afterwards).

    Parameters
    ----------
    array : numpy.ndarray
        Source array (2D label image or 3D RGB(A) image).
    zarr_path : str or os.PathLike
        Destination directory for the Zarr store.
    chunks : tuple of int, optional
        Chunk size along the leading (y, x) axes. Default (2048, 2048).
    overwrite : bool, optional
        Overwrite an existing store at ``zarr_path``. Default True.

    Returns
    -------
    str
        ``zarr_path``, as a string, for convenient chaining into the various
        ``open_*``/``*_dataarray`` helpers.
    """

    chunk_shape = _chunk_shape(array.shape, chunks)
    z = zarr.create_array(
        store=str(zarr_path), shape=array.shape, dtype=array.dtype,
        chunks=chunk_shape, overwrite=overwrite,
    )
    step = chunk_shape[0]
    for y0 in range(0, array.shape[0], step):
        y1 = min(y0 + step, array.shape[0])
        z[y0:y1] = array[y0:y1]
    return str(zarr_path)


def zeros_zarr(shape, zarr_path, dtype=np.uint8, chunks=DEFAULT_CHUNKS, overwrite=True):
    """
    Create an all-zero, on-disk Zarr store of the given shape/dtype, without
    ever allocating a full-size zero array in RAM.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array to create.
    zarr_path : str or os.PathLike
        Destination directory for the Zarr store.
    dtype : numpy dtype, optional
        Default ``numpy.uint8``.
    chunks : tuple of int, optional
        Chunk size along the leading (y, x) axes. Default (2048, 2048).
    overwrite : bool, optional
        Overwrite an existing store at ``zarr_path``. Default True.

    Returns
    -------
    str
        ``zarr_path``, as a string.
    """

    chunk_shape = _chunk_shape(shape, chunks)
    zarr.create_array(
        store=str(zarr_path), shape=shape, dtype=dtype,
        chunks=chunk_shape, fill_value=0, overwrite=overwrite,
    )
    return str(zarr_path)


def open_zarr_readonly(zarr_path, chunks='auto'):
    """Return a lazy, read-only ``dask.array`` view of a Zarr store on disk."""
    return da.from_zarr(str(zarr_path), chunks=chunks)


def image_dataarray(zarr_path, chunks='auto'):
    """
    Build a lazy, dask-backed ``xarray.DataArray`` view of an RGB(A) Zarr
    store for direct consumption by ``hv.RGB``.

    Parameters
    ----------
    zarr_path : str or os.PathLike
        Path to an on-disk Zarr store with shape ``(y, x, band)``.
    chunks : optional
        Dask chunking to use for the view. Default 'auto' (use the Zarr
        store's own on-disk chunking).

    Returns
    -------
    xarray.DataArray
        Dims ``(y, x, band)``, band coordinate labelled R/G/B(/A).
    """

    arr = open_zarr_readonly(zarr_path, chunks=chunks)
    band_names = BAND_NAMES[:arr.shape[-1]]
    # +0.5 so holoviews (which treats these as cell centres and pads by half a
    # cell on each side to infer plot bounds) reproduces exactly the same
    # (0, 0, w, h) pixel-edge bounds as the numpy `bounds=` path, keeping
    # pixel<->array-index mapping (and therefore drawn-stroke coordinates)
    # identical between the in-memory and file-backed code paths.
    coords = {
        'y': np.arange(arr.shape[0]) + 0.5, 'x': np.arange(arr.shape[1]) + 0.5, 'band': band_names,
    }
    return xr.DataArray(arr, dims=['y', 'x', 'band'], coords=coords)


def label_dataarray(zarr_path_or_array, chunks='auto'):
    """
    Build a lazy, dask-backed ``xarray.DataArray`` view of a 2D label Zarr
    store for direct consumption by ``hv.Image``.

    Because dask reads are deferred until ``.compute()``, a view built from
    a live (currently-open) writable array always reflects the latest writes
    made to that store -- there is no need to rebuild the view after each
    edit, only to re-push it through the rendering Pipe so holoviews knows
    to redraw.

    Parameters
    ----------
    zarr_path_or_array : str, os.PathLike, or zarr.Array
        Either a path to an on-disk Zarr store, or an already-open
        ``zarr.Array`` (e.g. from :class:`WritableLabelStore`).
    chunks : optional
        Dask chunking to use for the view. Default 'auto'.

    Returns
    -------
    xarray.DataArray
        Dims ``(y, x)``.
    """

    if isinstance(zarr_path_or_array, (str, os.PathLike)):
        arr = open_zarr_readonly(zarr_path_or_array, chunks=chunks)
    else:
        arr = da.from_zarr(zarr_path_or_array, chunks=chunks)
    # See the matching comment in image_dataarray() re: the +0.5 offset.
    coords = {'y': np.arange(arr.shape[0]) + 0.5, 'x': np.arange(arr.shape[1]) + 0.5}
    return xr.DataArray(arr, dims=['y', 'x'], coords=coords)


def tiff_to_zarr_store(tiff_path, level=0):
    """
    Open a (optionally pyramidal) TIFF/OME-TIFF lazily as a Zarr store,
    without decoding any pixel data up front. This is the genuinely
    zero-copy ingestion path (as opposed to :func:`array_to_zarr`, which
    still requires the source array to have been decoded into memory once
    by the caller, e.g. via PIL for PNG/JPEG which have no native chunked
    reader).

    Parameters
    ----------
    tiff_path : str or os.PathLike
        Path to the (OME-)TIFF file.
    level : int, optional
        Pyramid level to open, if the TIFF is pyramidal. Default 0 (full
        resolution).

    Returns
    -------
    zarr.Array
        Read-only, lazily-decoded view backed directly by the TIFF file.
    """

    import tifffile

    store = tifffile.imread(str(tiff_path), aszarr=True, level=level)
    return zarr.open_array(store=store, mode='r')


class WritableLabelStore:
    """
    Thin wrapper around a writable, on-disk Zarr label array, giving the
    annotator/segmenter UI bbox-scoped read/write access so committing a
    drawn stroke never requires materialising (or copying) the full label
    image in RAM -- only the bounding box actually touched by that stroke.

    Parameters
    ----------
    zarr_path : str or os.PathLike
        Path to an existing Zarr store (see :func:`zeros_zarr` /
        :func:`array_to_zarr` to create one).
    """

    def __init__(self, zarr_path):
        self.path = str(zarr_path)
        self._array = zarr.open_array(self.path, mode='r+')

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def chunk_shape(self):
        """On-disk chunk shape (y, x), e.g. for grouping many small writes so each on-disk
        chunk is touched once instead of once per write (see write_disks_batched)."""
        return self._array.chunks[:2]

    def read_block(self, y0, y1, x0, x1):
        """Read and return (as a small in-memory numpy array) just the
        ``[y0:y1, x0:x1]`` region."""
        return np.asarray(self._array[y0:y1, x0:x1])

    def write_block(self, y0, y1, x0, x1, block):
        """Write ``block`` into the ``[y0:y1, x0:x1]`` region in place, on
        disk."""
        self._array[y0:y1, x0:x1] = block

    def write_masked(self, y0, y1, x0, x1, local_mask, value, preserve_existing=False):
        """
        Write ``value`` into the pixels selected by ``local_mask`` (boolean array shaped like
        the ``[y0:y1, x0:x1]`` block) within that bounding box, leaving the rest of the block
        untouched. This is the shared primitive behind both the interactive annotator/segmenter's
        stroke commits and the sparse disk-shaped writes used for gene/background label
        assignment -- in both cases the caller only knows the shape of *one* touched region, not
        the whole array.

        Parameters
        ----------
        y0, y1, x0, x1 : int
            Bounding box of the block to read/write.
        local_mask : numpy.ndarray of bool
            Shape ``(y1 - y0, x1 - x0)``. Pixels where this is True are candidates for being set
            to ``value``.
        value : int
            Label value to write.
        preserve_existing : bool, optional
            If True, pixels that are already non-zero are left alone -- only pixels that are
            both selected by ``local_mask`` and currently 0 get written. Used when a "background"
            or other lower-priority label must never clobber an existing annotation. Default False
            (unconditional overwrite within ``local_mask``, e.g. for a higher-priority label that
            should always win).

        Returns
        -------
        numpy.ndarray
            The block's pre-write contents, for undo.
        """

        prev_block = self.read_block(y0, y1, x0, x1)
        write_mask = local_mask & (prev_block == 0) if preserve_existing else local_mask
        new_block = prev_block.copy()
        new_block[write_mask] = value
        self.write_block(y0, y1, x0, x1, new_block)
        return prev_block

    def to_dataarray(self, chunks='auto'):
        """Fresh lazy dask/xarray view of the store's current contents, for
        pushing through the rendering Pipe after a write."""
        return label_dataarray(self._array, chunks=chunks)

    def materialize(self):
        """Read the entire store into memory as a plain numpy array. Used
        only by the numpy-only consumers (classifier, median filter, ...)
        that have not been made chunk-aware; see ``annotation._ensure_in_memory``."""
        return np.asarray(self._array[:])
