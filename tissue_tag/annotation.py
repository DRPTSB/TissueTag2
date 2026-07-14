import base64
import copy as cp
import json
import logging
import tempfile
import warnings
from functools import partial
from io import BytesIO
from collections import OrderedDict

import bokeh
import datashader as ds
import holoviews as hv
import matplotlib.font_manager as fm
import numpy as np
import panel as pn
import pandas as pd
import param
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageColor
from bokeh.models import CustomJSHover, FreehandDrawTool, HoverTool, PolyDrawTool
from holoviews.operation import datashader as hd
from holoviews.streams import Pipe
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from packaging import version
from skimage import feature, future
from skimage.draw import polygon, disk
from skimage.future import trainable_segmentation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from scanpy.preprocessing import normalize_total
from tinybrain import downsample_segmentation

from tissue_tag.io import TissueTagAnnotation
from tissue_tag.organaxis import get_annotations_for_objects

hv.extension('bokeh')

# Holoviews/bokeh custom classes and functions
font_path = fm.findfont('DejaVu Sans')

# The native label-rendering path (see label_overlay) has been checked against
# holoviews 1.22.0 and datashader 0.18.2. It relies on:
#   - regrid accepting a string aggregator  (holoviews >= 1.14, AggregationOperation._agg_methods)
#   - regrid exposing `upsample` and `interpolation='nearest'`
#   - datashader Canvas.raster supporting mode/max/min/first/last resampling reductions
MIN_HOLOVIEWS = '1.14.0'
MIN_DATASHADER = '0.6.0'

# Fully transparent colour used for label value 0 (unannotated background) and NaN.
TRANSPARENT = 'rgba(0, 0, 0, 0)'

# Default dtype used when a new label_image is created from scratch.
DEFAULT_LABEL_DTYPE = np.uint8

# Fallback maximum number of distinct labels, used only when no label_image dtype is available
# yet. Once a label_image exists, its dtype's actual range (via np.iinfo) is used instead.
MAX_LABELS = np.iinfo(DEFAULT_LABEL_DTYPE).max

# Colours cycled through when the segmenter creates new objects.
SEGMENTER_COLORPOOL = ['green', 'cyan', 'brown', 'magenta', 'blue', 'red', 'orange']

# Per-task memory budget (bytes), used to patch datashader's Canvas.raster() (see below) so
# that regridding a file-backed (dask/Zarr-backed) image or label overlay stays low-RAM.
REGRID_MAX_MEM_BYTES = 64 * 1024 * 1024  # 64MB

_original_canvas_raster = ds.Canvas.raster


def _low_ram_canvas_raster(self, source, *args, **kwargs):
    """
    Wraps `datashader.Canvas.raster` to default `max_mem` (see `REGRID_MAX_MEM_BYTES`) whenever
    the caller doesn't specify one -- in particular, holoviews' `regrid` operation, which
    `label_image_overlay`/`base_image_element` use for file-backed rendering, never passes
    `max_mem`/`chunksize` itself (checked against holoviews 1.22.0).

    Why this matters: without a `max_mem` budget, datashader's `compute_chunksize` (see
    `datashader.resampling`) falls back to using the source array's *own* dask chunksize
    verbatim as the OUTPUT-space chunk grid. Our on-disk Zarr chunks are ~2048px
    (`file_backed.DEFAULT_CHUNKS`), far larger than a typical downsampled overview (a few
    hundred px) -- so that fallback collapses the "chunked" resample into a single task
    spanning the *entire* source array, silently pulling a whole 35000x35000 image into RAM
    regardless of viewport/zoom (confirmed by measuring resident memory directly; the resulting
    *displayed* image is still correctly downsampled, but getting there materialised everything
    once). Passing `max_mem` switches datashader onto its adaptive path instead, which derives a
    chunk size fine enough that each task's corresponding input region provably stays under the
    budget, regardless of how the source array happens to be chunked on disk or how large the
    downsampling ratio is.

    Harmless for plain-numpy (non-dask) sources -- `max_mem` is only consulted on the dask-array
    branch of `raster()` -- and for callers that already pass their own `max_mem`/`chunksize`.
    """

    kwargs.setdefault('max_mem', REGRID_MAX_MEM_BYTES)
    return _original_canvas_raster(self, source, *args, **kwargs)


ds.Canvas.raster = _low_ram_canvas_raster


class CustomFreehandDraw(hv.streams.FreehandDraw):
    """
    This custom class adds the ability to customise the icon for the FreeHandDraw tool.
    """

    def __init__(self, empty_value=None, num_objects=0, styles=None, tooltip=None, icon_colour="black", **params):
        self.icon_colour = icon_colour
        super().__init__(empty_value, num_objects, styles, tooltip, **params)


class CustomFreehandDrawCallback(hv.plotting.bokeh.callbacks.PolyDrawCallback):
    """
    This custom class is the corresponding callback for the CustomFreeHandDraw which will render a custom icon for
    the FreeHandDraw tool.
    """

    def initialize(self, plot_id=None):
        plot = self.plot
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        stream = self.streams[0]
        if stream.styles:
            self._create_style_callback(cds, glyph)
        kwargs = {}
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        if stream.empty_value is not None:
            kwargs['empty_value'] = stream.empty_value
        kwargs['icon'] = create_icon(stream.tooltip[0], stream.icon_colour)
        poly_tool = FreehandDrawTool(
            num_objects=stream.num_objects,
            renderers=[plot.handles['glyph_renderer']],
            **kwargs
        )
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims(cds.data)
        hv.plotting.bokeh.callbacks.CDSCallback.initialize(self, plot_id)


class CustomPolyDraw(hv.streams.PolyDraw):
    """
    This custom class adds the ability to customise the icon for the PolyDraw tool.
    """

    def __init__(self, empty_value=None, drag=True, num_objects=0, show_vertices=False, vertex_style={}, styles={},
                 tooltip=None, icon_colour="black", **params):
        self.icon_colour = icon_colour
        super().__init__(empty_value, drag, num_objects, show_vertices, vertex_style, styles, tooltip, **params)


class CustomPolyDrawCallback(hv.plotting.bokeh.callbacks.GlyphDrawCallback):
    """
    This custom class is the corresponding callback for the CustomPolyDraw which will render a custom icon for
    the PolyDraw tool.
    """

    def initialize(self, plot_id=None):
        plot = self.plot
        stream = self.streams[0]
        cds = self.plot.handles['cds']
        glyph = self.plot.handles['glyph']
        renderers = [plot.handles['glyph_renderer']]
        kwargs = {}
        if stream.num_objects:
            kwargs['num_objects'] = stream.num_objects
        if stream.show_vertices:
            vertex_style = dict({'size': 10}, **stream.vertex_style)
            r1 = plot.state.scatter([], [], **vertex_style)
            kwargs['vertex_renderer'] = r1
        if stream.styles:
            self._create_style_callback(cds, glyph)
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        if stream.empty_value is not None:
            kwargs['empty_value'] = stream.empty_value
        kwargs['icon'] = 'delete'
        poly_tool = PolyDrawTool(
            drag=all(s.drag for s in self.streams), renderers=renderers,
            **kwargs
        )
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims(cds.data)
        super().initialize(plot_id)


# Overload the callback from holoviews to use the custom FreeHandDrawCallback class. Probably not safe.
hv.plotting.bokeh.callbacks.Stream._callbacks['bokeh'].update({
    CustomFreehandDraw: CustomFreehandDrawCallback,
    CustomPolyDraw: CustomPolyDrawCallback
})


_original_cds_callback_initialize = hv.plotting.bokeh.callbacks.CDSCallback.initialize

def _custom_cds_callback_initializer(self, plot_id=None):
    """
    This custom initialiser for CDSCallback (the common base class for every draw-tool callback)
    allows each stream to keep a reference to the bokeh ColumnDataSource backing its glyph so we
    can clear it from the backend.
    """

    _original_cds_callback_initialize(self, plot_id)
    cds = self.plot.handles.get('source')
    if cds is not None:
        for stream in self.streams:
            stream._draw_cds = cds


# Overload CDSCallback.initialize to use the custom initialiser.
hv.plotting.bokeh.callbacks.CDSCallback.initialize = _custom_cds_callback_initializer


def to_base64(img):
    """
    Helper function to convert an Image object to base64 string.

    Parameters
    ----------
    img: PIL.Image
        PIL Image object to convert to base64 string.

    Returns
    -------
    str
        Base64 string of the image in PNG format.
    """

    buffered = BytesIO()
    img.save(buffered, format="png")
    data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{data}'


def create_icon(name, colour):
    """
    Helper function to create a custom icon composed on a coloured rectangle with a letter in the middle.

    Parameters
    ----------
    name: str
        Name of the tool.
    colour: str
        Color of the icon. Can be a color name or an RGB tuple.

    Returns
    -------
    PIL.Image
        PIL Image object of the icon.
    """

    font_size = 24
    size = (30, 30)

    if isinstance(colour, str):
        # Create a temporary 1x1 image to convert color name to RGB
        temp_img = Image.new("RGB", (1, 1), colour)
        colour = temp_img.getpixel((0, 0))
    else:
        colour = colour

    # Create a new image with the specified background color
    image = Image.new('RGB', size, color=colour)
    draw = ImageDraw.Draw(image)

    # Calculate font size if not provided
    if font_size is None:
        font_size = int(min(size) * 0.4)

    # Try to use the specified font, fallback to default if not available
    font = ImageFont.truetype(font_path, font_size)

    # Calculate text position to center it
    # For newer Pillow versions use textbbox
    try:
        # Get the bounding box of the text
        left, top, right, bottom = draw.textbbox((0, 0), name, font=font)
        text_width = right - left
        text_height = bottom - top

        # For better vertical centering with both uppercase and lowercase letters
        # Get metrics for a reference that includes both ascenders and descenders
        ref_left, ref_top, ref_right, ref_bottom = draw.textbbox((0, 0), "Ájpq", font=font)
        ref_height = ref_bottom - ref_top

        # Calculate visual center offset
        ascent_ratio = 0.8  # Approximate ratio for visual center (may need adjustment)

        # Calculate horizontal and vertical positions
        text_x = (size[0] - text_width) // 2 - left  # Adjust for left offset
        text_y = (size[1] - ref_height) // 2 + (ref_height * (1 - ascent_ratio)) - top

    except AttributeError:
        # Fall back to textsize for older Pillow versions
        text_width, text_height = draw.textsize(name, font=font)
        # Get the offset for proper alignment
        try:
            offset_x, offset_y = font.getoffset(name)
            # Get metrics for reference string
            ref_width, ref_height = draw.textsize("Ájpq", font=font)

            # Calculate positions with adjustment for visual center
            text_x = (size[0] - text_width) // 2 - offset_x
            text_y = (size[1] - ref_height) // 2 + (ref_height * 0.3) - offset_y
        except (AttributeError, TypeError):
            # Simplest fallback if getoffset is not available
            text_x = (size[0] - text_width) // 2
            text_y = (size[1] - text_height) // 2

    brightness = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]

    # Use white text on dark backgrounds, black text on light backgrounds
    text_color = "white" if brightness < 128 else "black"

    # Draw the letter
    draw.text((text_x, text_y), name, fill=text_color, font=font, stroke_width=0.2, stroke_fill=text_color)

    if version.parse(bokeh.__version__) < version.parse("3.1.0"):
        image = to_base64(image)
    return image


def hex_palette_generator(annotation_map):
    """
    Helper function to generate a list of hex colours from annotation_map values.

    Parameters
    ----------
    annotation_map: dict
        Mapping of annotation name to colour.

    Returns
    -------
    list of str
        List of "#rrggbb" hex colours.
    """

    palette = []
    for colour in annotation_map.values():
        r, g, b = ImageColor.getrgb(colour)[:3]
        palette.append(f'#{r:02x}{g:02x}{b:02x}')
    return palette


def annotation_label_hover_tool(label_names):
    """
    Custom HoverTool to show annotation name instead of the integer label value in label_image.

    Parameters
    ----------
    label_names: list of str
        Names of the labels in ``annotation_map`` order, so that ``label_names[i]`` is the name
        of the label with value ``i + 1``. Value 0 (special value) is always mapped to "background".

    Returns
    -------
    bokeh.models.HoverTool
        Hover tool whose tooltip resolves the underlying integer via a small JS lookup table,
        so it reads e.g. "Annotation: cortex" rather than "image: 3", alongside the x/y
        coordinates (matching the default 'hover' tool's tooltip).
    """

    names_json = json.dumps(['background'] + list(label_names))
    formatter = CustomJSHover(code=f"""
        var names = {names_json};
        var v = Math.round(value);
        if (v < 0 || v >= names.length) {{ return 'unknown'; }}
        return names[v];
    """)
    return HoverTool(
        tooltips=[('x', '$x'), ('y', '$y'), ('Annotation', '@image{custom}')],
        formatters={'@image': formatter},
    )


def build_annotation_toggle_widget(annotation_map):
    """
    Helper function to build a custom widget to allow user to toggle visibility of individual annotations
    from ``annotation_map`` in the rendered overlay.

    Parameters
    ----------
    annotation_map: dict
        Mapping of annotation name to colour.

    Returns
    -------
    state: param.Parameterized
        Object whose ``value`` List parameter holds the currently visible label names. Pass this
        as the ``visibility_widget`` argument to ``label_overlay``.
    widget_ui: panel.FlexBox
        The "Annotations:" label, a "Select all" checkbox that shows/hides every label at once,
        followed by one colour-swatch + checkbox per toggle-able label. The "Select all" checkbox
        also reflects the aggregate state (checked only when every label is currently visible).
    """

    class _AnnotationVisibilityState(param.Parameterized):
        """
        Custom param object exposing the list of currently visible label names, replicating how the
        ``pn.widgets.CheckBoxGroup`` works for defining what is visible.
        """

        value = param.List(default=[])

    checkboxes = {}
    select_all_toggle = pn.widgets.Checkbox(value=True, name='Select all', width=90)
    items = [
        pn.widgets.StaticText(value='Annotations:', styles={'font-weight': 'bold'}),
        select_all_toggle,
    ]

    state = _AnnotationVisibilityState(value=list(annotation_map.keys()))

    _updating_toggle = [False]  # guard against recursive callbacks between toggle <-> checkboxes
    _bulk_update = [False]  # guard to skip per-checkbox _refresh work during "Select all" toggling

    def _refresh(event=None):
        if _bulk_update[0]:
            return
        state.value = [name for name, cb in checkboxes.items() if cb.value]
        if not _updating_toggle[0]:
            _updating_toggle[0] = True
            select_all_toggle.value = all(cb.value for cb in checkboxes.values())
            _updating_toggle[0] = False

    def _toggle_all(event):
        if _updating_toggle[0]:
            return
        _updating_toggle[0] = True
        _bulk_update[0] = True

        # Batch all checkbox updates into a single document/websocket push instead of one
        # per checkbox, which is what caused the visible sequential toggling with a delay.
        # `hold` combines the underlying Bokeh document events, while `block_comm` stops each
        # checkbox from eagerly pushing its own change over the notebook comm; `push_notebook`
        # then flushes everything as one message so all checkboxes (un)tick simultaneously.
        checkbox_list = list(checkboxes.values())
        with pn.io.hold(pn.state.curdoc), pn.io.block_comm():
            for cb in checkbox_list:
                cb.value = event.new
        if checkbox_list:
            pn.io.push_notebook(*checkbox_list)
        _bulk_update[0] = False
        _updating_toggle[0] = False
        _refresh()

    select_all_toggle.param.watch(_toggle_all, 'value')

    for name, colour in annotation_map.items():
        cb = pn.widgets.Checkbox(value=True, name='', width=18)
        swatch = pn.pane.HTML(
            f'<div style="width:14px; height:14px; background-color:{colour}; '
            'border:1px solid rgba(0,0,0,0.4); border-radius:3px;"></div>',
            width=16, height=16, margin=(2, 4, 0, 0),
        )
        label = pn.widgets.StaticText(value=name, margin=(4, 8, 0, 0))
        checkboxes[name] = cb
        cb.param.watch(_refresh, 'value')
        items.append(pn.Row(cb, swatch, label, margin=(0, 6, 0, 0)))

    _refresh()
    ui = pn.FlexBox(*items, align_items='center')
    return state, ui


def _is_xarray_backed(data):
    """
    Whether `data` is a lazy, dask-backed xarray.DataArray (i.e. produced by
    `tissue_tag.file_backed`) rather than a plain in-memory numpy array.
    """

    return hasattr(data, 'dims') and hasattr(data, 'coords')


def label_image_element(data, invert_y=False):
    """
    Helper function to wrap a 2D label_image into an hv.Image

    Parameters
    ----------
    data: numpy.ndarray or xarray.DataArray
        2D integer array of label values. May be a lazy, dask-backed
        xarray.DataArray (see `tissue_tag.file_backed`), in which case the
        flip is a cheap lazy re-index and no full-resolution copy is made.
    invert_y: bool, optional
        Invert plot along y axis. Default is False.

    Returns
    -------
    holoviews.Image
        Image element with a single 'label' value dimension.
    """

    if _is_xarray_backed(data):
        arr = data if invert_y else data[::-1]
        return hv.Image(arr, kdims=['x', 'y'], vdims=['label'])

    arr = data if invert_y else np.ascontiguousarray(data[::-1])
    h, w = arr.shape
    return hv.Image(arr, bounds=(0, 0, w, h), vdims=['label'])


def label_image_overlay(label_pipe, palette, plot_size=1024, invert_y=False,
                        use_datashader=False, annotation_aggregator='max', opacity_param_object=None,
                        annotation_toggle_param_object=None, annotation_names=None):
    """
    Build the dynamic, natively-coloured annotation overlay.

    The overlay is driven by a Pipe stream carrying the current label_image, so pushing a new
    label_image redraws the annotation *without* rebuilding the bokeh plot. This preserves the
    user's zoom/pan across updates.

    Parameters
    ----------
    label_pipe: holoviews.streams.Pipe
        Pipe carrying the current 2D integer label_image.
    palette: list of str
        Hex colours, where palette[i] is the colour for annotation value i + 1. The number of
        distinct (non-background) annotations the palette covers is taken as ``len(palette)``.
    plot_size: int, optional
        Figure size for plotting. Default is 1024.
    invert_y: bool, optional
        Invert plot along y axis. Default is False.
    use_datashader: bool, optional
        Use datashader regridding. Recommended for high resolution images. Default is False.
    annotation_aggregator: str, optional
        Datashader reduction used when several label pixels fall into one screen pixel.
        Only used when use_datashader is True. Default is 'max'.

        - 'max'  : any labelled pixel beats background, so thin strokes stay visible when
                   zoomed out. Ties within a screen pixel resolve to the highest label value.
        - 'mode' : majority label wins, which is area-accurate but makes sparse strokes
                   disappear at low zoom (they lose the vote to background).
        - 'first' / 'last' / 'min' are also accepted.

        NOTE: do NOT use the datashader default of 'mean'. Averaging label 1 and label 3 gives
        label 2, which invents classes that were never annotated.
    opacity_param_object: panel.widgets.FloatSlider, optional
        Slider whose value drives the overlay alpha. Default is None.
    annotation_toggle_param_object: panel.widgets.CheckBoxGroup, optional
        Checkbox group whose ``value`` controls which annotations are shown. Excluded annotations
        are masked out of the rendered overlay (their pixels are treated as background/transparent)
        without touching the underlying label_image. Must be used together with ``annotation_names``.
        Default is None (all annotations always shown).
    annotation_names: list of str, optional`
        Names of the annotations in ``annotation_map`` order, so that ``annotation_names[i]`` is the name
        of the annotation with value ``i + 1``. Required if ``annotation_toggle_param_object`` is given.

    Returns
    -------
    holoviews.DynamicMap
        The annotation overlay, ready to be composed into an Overlay.
    """

    n_annotations = len(palette)

    valid_aggregators = ('max', 'mode', 'min', 'first', 'last')
    if annotation_aggregator not in valid_aggregators:
        raise ValueError(
            f"label_aggregator must be one of {valid_aggregators!r}, got {annotation_aggregator!r}. "
            "In particular 'mean' (the datashader default) is invalid here: averaging label "
            "values produces classes that were never annotated, e.g. mean(1, 3) == 2."
        )

    if annotation_toggle_param_object is not None and annotation_names is not None:
        name_to_value = {name: idx + 1 for idx, name in enumerate(annotation_names)}

        def mask_hidden_labels(data, value):
            visible_names = annotation_names if value is None else value
            hidden_values = [name_to_value[name] for name in annotation_names if name not in visible_names]
            if hidden_values:
                lut = np.arange(np.iinfo(data.dtype).max + 1, dtype=data.dtype)
                lut[hidden_values] = 0
                if _is_xarray_backed(data):
                    # Apply the LUT chunk-by-chunk so masking a file-backed
                    # label image never materialises the full array in RAM.
                    import xarray as xr
                    masked = data.data.map_blocks(lambda block: lut[block], dtype=data.dtype)
                    data = xr.DataArray(masked, dims=data.dims, coords=data.coords)
                else:
                    data = lut[data]
            return label_image_element(data, invert_y=invert_y)

        anno = hv.DynamicMap(
            mask_hidden_labels,
            streams=[label_pipe, hv.streams.Params(parameters=[annotation_toggle_param_object.param.value])]
        )
    else:
        anno = hv.DynamicMap(partial(label_image_element, invert_y=invert_y), streams=[label_pipe])

    if use_datashader:
        anno = hd.regrid(
            anno,
            aggregator=annotation_aggregator,
            interpolation='nearest',   # never blend label values when upsampling
            upsample=True,
        )

    anno = anno.opts(
        cmap=palette,
        # Bin edges land on the half-integers, so integer label value v maps to palette[v - 1].
        clim=(0.5, n_annotations + 0.5),
        # Discrete bins: stops the colormapper interpolating between neighbouring label colours.
        color_levels=n_annotations,
        # Label 0 (background) falls below the clim floor and is clipped to fully transparent.
        clipping_colors={'min': TRANSPARENT, 'NaN': TRANSPARENT},
        colorbar=False,
        aspect='equal',
        frame_height=int(plot_size),
        frame_width=int(plot_size),
        tools=[annotation_label_hover_tool(annotation_names)] if annotation_names is not None else ['hover'],
        backend_opts={"plot.toolbar_location": "left"},
    )

    if opacity_param_object is not None:
        anno = anno.apply.opts(alpha=opacity_param_object.param.value)

    return anno


def base_image_element(image, plot_size=1024, invert_y=False, use_datashader=False):
    """
    Helper function to build the static base (morphology) image layer.

    Unlike the previous implementation this is constructed exactly once, since the base
    image never changes during annotation.

    Parameters
    ----------
    image: numpy.ndarray or xarray.DataArray
        RGB(A) base image. May be a lazy, dask-backed xarray.DataArray (see
        `tissue_tag.file_backed`), in which case this function never
        materialises the full-resolution image; `regrid`/datashader only
        pulls in the pixels needed for the current viewport and zoom level.
    plot_size: int, optional
        Figure size for plotting. Default is 1024.
    invert_y: bool, optional
        Invert plot along y axis. Default is False.
    use_datashader: bool, optional
        Use datashader regridding. Default is False.

    Returns
    -------
    holoviews.RGB | holoviews.DynamicMap
        The base image layer.
    """

    if _is_xarray_backed(image):
        imarray_c = image if image.dtype == np.uint8 else image.astype('uint8')
        if not invert_y:
            imarray_c = imarray_c[::-1]
        img = hv.RGB(imarray_c, kdims=['x', 'y'], vdims=list(imarray_c.coords['band'].values))
    else:
        imarray_c = image.astype('uint8')
        if not invert_y:
            imarray_c = np.flip(imarray_c, 0)
        img = hv.RGB(imarray_c, bounds=(0, 0, imarray_c.shape[1], imarray_c.shape[0]))

    if use_datashader:
        img = hd.regrid(img)

    return img.opts(aspect="equal", frame_height=int(plot_size), frame_width=int(plot_size))


def clear_draw_stream(stream):
    """
    Helper function to clear the strokes of a draw-tool stream after they have been committed to the label image.

    Since the bokeh plot is not being rebuilt after update and reset, the strokes remains on the bokeh canvas.
    This fixes the issue by clearing the ColumnDataSource backing the glyph (``_draw_cds``).

    Parameters
    ----------
    stream: holoviews.streams.Stream
        A FreehandDraw / PolyDraw stream.

    Returns
    -------
    None
    """

    cds = getattr(stream, '_draw_cds', None)
    if cds is not None:
        try:
            cds.data = {column: [] for column in cds.data}
        except Exception as err:  # pragma: no cover - depends on holoviews/bokeh version
            warnings.warn(f"Could not clear the bokeh ColumnDataSource directly ({err}); strokes "
                          "may remain visible on the canvas until removed manually.")

    try:
        stream.event(data={'xs': [], 'ys': []})
    except Exception as err:  # pragma: no cover - depends on holoviews/bokeh version
        warnings.warn(f"Could not reset draw stream ({err}); committed strokes will be re-applied "
                      "idempotently on the next update.")


def _write_polygon_strokes_file_backed(writer, strokes):
    """
    Commit a batch of drawn polygon strokes directly onto an on-disk label
    store, one bounding-box-scoped read/write per stroke, so committing an
    Update never requires materialising (or copying) the full label image in
    RAM -- only the small region each stroke actually touches.

    Parameters
    ----------
    writer: file_backed.WritableLabelStore
        Writable handle onto the on-disk label Zarr store.
    strokes: list of (xs, ys, label_value)
        One entry per drawn stroke: the bokeh draw-tool's raw x/y vertex
        lists and the label value to paint that stroke's interior with (0
        for an eraser stroke).

    Returns
    -------
    list of (y0, y1, x0, x1, previous_block)
        The pre-write contents of every block actually written, in write
        order -- pass to :func:`_revert_polygon_strokes_file_backed` (in
        reverse) to undo exactly this batch.
    """

    written = []
    for xs, ys, label_value in strokes:
        x = np.array(xs).astype(int)
        y = np.array(ys).astype(int)
        rr, cc = polygon(y, x)
        inshape = (writer.shape[0] > rr) & (0 < rr) & (writer.shape[1] > cc) & (0 < cc)
        rr_in, cc_in = rr[inshape], cc[inshape]
        if rr_in.size == 0:
            continue

        y0, y1 = int(rr_in.min()), int(rr_in.max()) + 1
        x0, x1 = int(cc_in.min()), int(cc_in.max()) + 1

        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        local_mask[rr_in - y0, cc_in - x0] = True
        prev_block = writer.write_masked(y0, y1, x0, x1, local_mask, label_value, preserve_existing=False)
        written.append((y0, y1, x0, x1, prev_block))

    return written


def _revert_polygon_strokes_file_backed(writer, written_blocks):
    """Undo a batch produced by :func:`_write_polygon_strokes_file_backed`,
    restoring each touched block in reverse write order."""

    for y0, y1, x0, x1, block in reversed(written_blocks):
        writer.write_block(y0, y1, x0, x1, block)


# Annotation functions

def annotator(tissue_tag_annotation, plot_size=1024, invert_y=False, use_datashader=False,
              unassigned_colour="yellow", annotation_aggregator='max', clear_paths_on_update=True,
              file_backed=False, work_dir=None):
    """
    Interactive annotation tool with line annotations using Panel for switching between morphology and annotation.

    The annotation layer is rendered directly from ``label_image`` by holoviews/datashader, with
    colour applied as a plot option. ``rgb_from_labels`` is no longer called on the interactive
    path, which removes a full H x W x 4 array allocation from every update and lets the plot be
    updated in place (so zoom and pan now survive an Update).

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image and annotation map.
    plot_size: int, default=1024
        Figure size for plotting.
    invert_y : bool, optional
        invert plot along y axis
    use_datashader : bool, optional
        If we should use datashader for rendering the image. Recommended for high resolution image. Default is False.
    unassigned_colour : str, optional
        Color for unassigned pixels. Default is "yellow".
    annotation_aggregator : str, optional
        Reduction used to downsample the label image when zoomed out. See ``label_image_overlay``.
        Default is 'max', which keeps thin strokes visible at low zoom.
    clear_paths_on_update : bool, optional
        Clear the drawn strokes once they have been committed to the label image. Default is True.
    file_backed : bool, optional
        Keep ``image``/``label_image`` on disk (Zarr, via ``tissue_tag.file_backed``) rather than
        fully in RAM. Rendering flows through datashader's ``regrid`` so only the current viewport
        is materialised, and each Update writes only the bounding box of the drawn strokes straight
        to the on-disk label store -- the process never holds a second full-resolution copy of
        either array. If ``tissue_tag_annotation`` is already file-backed (see
        ``TissueTagAnnotation.to_file_backed``) this is inferred automatically. Default is False.
    work_dir : str, optional
        Directory to hold the on-disk Zarr stores when ``file_backed`` triggers a fresh conversion.
        Defaults to a new temporary directory. Ignored if ``tissue_tag_annotation`` is already
        file-backed.

    Returns
    -------
    Panel app
        A panel application object with annotator tool.
    """

    logging.getLogger('bokeh.core.validation.check').setLevel(logging.ERROR)

    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide annotation map.")
    else:
        tissue_tag_annotation.annotation_map = OrderedDict(tissue_tag_annotation.annotation_map)
        tissue_tag_annotation.annotation_map["unassigned"] = unassigned_colour
        tissue_tag_annotation.annotation_map.move_to_end("unassigned", last=False)

    use_file_backed = file_backed or tissue_tag_annotation.file_backed
    if use_file_backed:
        if not tissue_tag_annotation.file_backed:
            tissue_tag_annotation.to_file_backed(work_dir or tempfile.mkdtemp())
        else:
            # to_file_backed() already calls this; also cover objects that were made
            # file-backed by direct field assignment rather than via that method.
            from tissue_tag import file_backed as fb
            fb.configure_dask_for_low_ram()
    elif tissue_tag_annotation.label_image is None:
        # An all-zero label image simply renders as nothing, so the previous
        # {'default': '#00000000'} placeholder annotation_map is no longer needed.
        tissue_tag_annotation.label_image = np.zeros(
            (tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]),
            dtype=DEFAULT_LABEL_DTYPE
        )

    palette = hex_palette_generator(tissue_tag_annotation.annotation_map)

    update_button = pn.widgets.Button(name='Update', button_type='primary')
    revert_button = pn.widgets.Button(name='Revert', button_type='danger', disabled=True)
    label_image_opacity = pn.widgets.FloatSlider(name='Label overlay', value=0.5, start=0, end=1, step=0.1)
    annotation_names = list(tissue_tag_annotation.annotation_map.keys())
    annotation_toggle_param, annotation_toggle_widget = build_annotation_toggle_widget(
        tissue_tag_annotation.annotation_map
    )

    # The label image is the single source of truth; the Pipe carries it to the plot.
    label_pipe = Pipe(data=tissue_tag_annotation.label_image)

    ds_img = base_image_element(tissue_tag_annotation.image, plot_size=plot_size,
                                invert_y=invert_y, use_datashader=use_datashader)
    ds_anno = label_image_overlay(label_pipe, palette, plot_size=plot_size, invert_y=invert_y,
                                  use_datashader=use_datashader, annotation_aggregator=annotation_aggregator,
                                  opacity_param_object=label_image_opacity, annotation_toggle_param_object=annotation_toggle_param,
                                  annotation_names=annotation_names)

    plot_list = [ds_img, ds_anno]

    render_dict = {}
    path_dict = {}
    for key in tissue_tag_annotation.annotation_map.keys():
        path_dict[key] = hv.Path([]).opts(color=tissue_tag_annotation.annotation_map[key], line_width=5, line_alpha=0.7)
        render_dict[key] = CustomFreehandDraw(source=path_dict[key], num_objects=200, tooltip=key,
                                              icon_colour=tissue_tag_annotation.annotation_map[key])

        plot_list.append(path_dict[key])

    # Built once. Never rebuilt, so the bokeh plot (and the user's zoom/pan) survives updates.
    tab_object = pn.panel(hv.Overlay(plot_list).collate())
    p = pn.Column(
        pn.Row(label_image_opacity, update_button, revert_button),
        annotation_toggle_widget,
        tab_object,
    )

    previous_labels = None if use_file_backed else tissue_tag_annotation.label_image.copy()
    previous_blocks = []  # file-backed only: (y0, y1, x0, x1, previous_block) from the last Update

    def update_annotator(event):
        nonlocal previous_labels, previous_blocks

        if not event:
            return

        update_button.disabled = True

        if use_file_backed:
            strokes = [
                (render_dict[a].data['xs'][o], render_dict[a].data['ys'][o], idx + 1)
                for idx, a in enumerate(render_dict.keys())
                if render_dict[a].data['xs']
                for o in range(len(render_dict[a].data['xs']))
            ]
            writer = tissue_tag_annotation.label_writer()
            previous_blocks = _write_polygon_strokes_file_backed(writer, strokes)
            tissue_tag_annotation.refresh_label_view()
            label_pipe.send(tissue_tag_annotation.label_image)
        else:
            previous_labels = tissue_tag_annotation.label_image.copy()
            # Work on a copy: param compares old/new values, so pushing the same array object back
            # through the Pipe may not fire a redraw.
            updated_labels = tissue_tag_annotation.label_image.copy()
            for idx, a in enumerate(render_dict.keys()):
                if render_dict[a].data['xs']:
                    for o in range(len(render_dict[a].data['xs'])):
                        x = np.array(render_dict[a].data['xs'][o]).astype(int)
                        y = np.array(render_dict[a].data['ys'][o]).astype(int)
                        rr, cc = polygon(y, x)
                        inshape = np.where(
                            np.array(tissue_tag_annotation.label_image.shape[0] > rr) & np.array(0 < rr) & np.array(
                                tissue_tag_annotation.label_image.shape[1] > cc) & np.array(
                                0 < cc))[0]
                        updated_labels[rr[inshape], cc[inshape]] = idx + 1

            tissue_tag_annotation.label_image = updated_labels

            # This single line replaces rgb_from_labels() + create_images() + pn.panel() rebuild.
            label_pipe.send(updated_labels)

        if clear_paths_on_update:
            for stream in render_dict.values():
                clear_draw_stream(stream)

        revert_button.disabled = False
        update_button.disabled = False

    def revert_annotator(event):
        if not event:
            return

        update_button.disabled = True

        if use_file_backed:
            writer = tissue_tag_annotation.label_writer()
            _revert_polygon_strokes_file_backed(writer, previous_blocks)
            tissue_tag_annotation.refresh_label_view()
            label_pipe.send(tissue_tag_annotation.label_image)
        else:
            tissue_tag_annotation.label_image = previous_labels.copy()
            label_pipe.send(tissue_tag_annotation.label_image)

        for stream in render_dict.values():
            clear_draw_stream(stream)

        revert_button.disabled = True
        update_button.disabled = False

    pn.bind(update_annotator, update_button, watch=True)
    pn.bind(revert_annotator, revert_button, watch=True)

    return p


def _ensure_in_memory(tissue_tag_annotation):
    """
    Materialize a file-backed ``TissueTagAnnotation``'s ``image``/``label_image`` into plain
    in-memory numpy arrays, in place, before handing off to a function that is not chunk-aware.

    This is a deliberate, documented scope boundary: the pixel classifier (feature extraction in
    particular), the median filter, and a couple of other numpy-only helpers have not been
    rewritten to operate chunk-by-chunk on dask/Zarr-backed data, so they still need the full
    array resident in RAM -- same as today's ``downsampling_factor`` option is used to manage
    that cost. Everything on the annotator/segmenter/viewing path stays low-RAM regardless; only
    these specific numpy-only operations fall back to materialising here. No-op if
    ``tissue_tag_annotation`` is not file-backed.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation

    Returns
    -------
    TissueTagAnnotation
        ``tissue_tag_annotation``, for chaining. Callers that want to preserve a file-backed
        original should pass a ``copy=True``'d object into this function (as every caller below
        does), not the caller's own reference.
    """

    if not tissue_tag_annotation.file_backed:
        return tissue_tag_annotation

    warnings.warn(
        "This operation is not chunk-aware and will load the full image/label_image into RAM "
        "(equivalent to the in-memory pipeline's peak usage for this step); see "
        "tissue_tag.file_backed and the file_backed docs on annotator/segmenter for the parts "
        "of the pipeline that stay low-RAM."
    )
    tissue_tag_annotation.image = np.asarray(tissue_tag_annotation.image)
    if tissue_tag_annotation.label_image is not None:
        tissue_tag_annotation.label_image = np.asarray(tissue_tag_annotation.label_image)
    tissue_tag_annotation.image_store = None
    tissue_tag_annotation.label_store = None
    return tissue_tag_annotation


def rgb_from_labels(tissue_tag_annotation):
    """
    Helper function to generate colored annotation image from label image and annotation map.

    NOTE: this is no longer used on the interactive annotator/segmenter path, which now renders
    label_image natively via holoviews/datashader. It is retained for static plotting
    (``plot_labels``, ``overlay_labels``) and for exporting a flattened RGBA annotation.

    The intermediate array is now allocated as uint8 rather than float64, which cuts peak memory
    for this function by 8x (a 20k x 20k label image previously allocated ~12.8 GB here).

    NOTE: not chunk-aware -- if ``tissue_tag_annotation`` is file-backed, this materialises the
    full label_image into RAM (see ``_ensure_in_memory``).

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with label_image and annotation map.

    Returns
    -------
    numpy.ndarray
        Annotation image.
    """

    # Read-only: materialise a local numpy view without mutating the caller's (possibly
    # file-backed) object, unlike the classifier/median_filter/... helpers below which already
    # have copy=False/True semantics that make an in-place materialisation expected.
    label_image = np.asarray(tissue_tag_annotation.label_image)

    labelimage_rgb = np.zeros(
        (label_image.shape[0], label_image.shape[1], 4),
        dtype=np.uint8
    )

    colours = list(tissue_tag_annotation.annotation_map.values())
    for c in range(len(colours)):
        color = ImageColor.getcolor(colours[c], "RGBA")
        labelimage_rgb[label_image == c + 1, 0:4] = np.array(color, dtype=np.uint8)

    return labelimage_rgb


def pixel_label_classifier(tissue_tag_annotation, classifier="RandomForest", threshold=None, downsampling_factor=1, plot=True, copy=False):
    """
    A pixel classifier using all RGB channels as features.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image, label_image and annotation map.
    classifier: str, optional
        Algorithm to use for classification. Currently supported algorithms are RandomForest and LogisticRegression from sklearn.
    threshold: float, optional
        Minimum probability score threshold for a pixel to be assigned a label by the random forest classifier.
    downsampling_factor: integer, optional
        Downsampling factor to resize image and label_image prior to running classifier. Default value of 1 means no downsampling is performed.
    plot: bool, optional
        Plot the resulting label_image after running the clasifier.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the label_image in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with updated label_image based on the classifier prediction if copy is True,
        otherwise None.

    Notes
    -----
    Known scope limitation: feature extraction (``skimage.feature.multiscale_basic_features``)
    is not chunk-aware, so this always materialises the (optionally downsampled) image/label_image
    fully in RAM, even for a file-backed ``tissue_tag_annotation`` (see ``_ensure_in_memory``).
    Use ``downsampling_factor`` to manage peak memory on very large images.
    """

    def predict_segmenter_thresholded(features, clf, threshold):
        sh = features.shape
        if features.ndim > 2:
            features = features.reshape((-1, sh[-1]))

        try:
            predicted_labels_probability = clf.predict_proba(features)
        except NotFittedError:
            raise NotFittedError(
                "You must train the classifier `clf` first"
                "for example with the `fit_segmenter` function."
            )
        except ValueError as err:
            if err.args and 'x must consist of vectors of length' in err.args[0]:
                raise ValueError(
                    err.args[0]
                    + '\n'
                    + "Maybe you did not use the same type of features for training the classifier."
                )
            else:
                raise err

        label_classes = np.concatenate((np.array([0]), clf.classes_))
        max_probability_indices = np.argmax(predicted_labels_probability, axis=1) + 1
        no_label_mask = np.max(predicted_labels_probability, axis=1) < threshold
        max_probability_indices[no_label_mask] = 0
        predicted_labels = label_classes[max_probability_indices]

        output = predicted_labels.reshape(sh[:-1])
        return output

    if classifier not in ["RandomForest", "LogisticRegression"]:
        raise ValueError("Classifier is not supported. Currently supported classifiers are RandomForest and LogisticRegression.")

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation
    tissue_tag_annotation = _ensure_in_memory(tissue_tag_annotation)

    print("[INFO] Initializing classifier...")
    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1)  # Process all channels together

    image = tissue_tag_annotation.image
    label_image = tissue_tag_annotation.label_image

    if downsampling_factor > 1:
        print("[INFO] Downsampling image and label_image...")
        label_image = downsample_segmentation(label_image, (downsampling_factor, downsampling_factor))[0]
        image = cv2.resize(image, label_image.shape[::-1], interpolation=cv2.INTER_LANCZOS4)
        print(f"[INFO]  - Original resolution: {tissue_tag_annotation.label_image.shape}. Downsampled resolution: {label_image.shape}")

    print("[INFO] Extracting features from all RGB channels...")
    features = features_func(image)  # Extract multiscale features for all channels at once

    del(image)

    print(f"[INFO] Training {classifier} classifier on RGB features...")
    if classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
    elif classifier == "LogisticRegression":
        clf = LogisticRegression(n_jobs=-1, random_state=42)
    clf = trainable_segmentation.fit_segmenter(label_image, features, clf)

    print("[INFO] Predicting labels based on trained classifier...")
    if threshold is not None:
        print(f"[INFO] - Using {threshold} probability threshold for assigning label.")
        predicted_labels = predict_segmenter_thresholded(features, clf, threshold)
    else:
        predicted_labels = trainable_segmentation.predict_segmenter(features, clf)

    if downsampling_factor > 1:
        print("[INFO] Upsampling predicted label_image...")
        predicted_labels = cv2.resize(predicted_labels, tissue_tag_annotation.label_image.shape[::-1], interpolation=cv2.INTER_NEAREST)

    print("[INFO] Final label prediction completed.")
    tissue_tag_annotation.label_image = predicted_labels

    del(label_image)
    del(predicted_labels)
    del(features)
    del(clf)

    if plot:
        print("[INFO] Generating visualization...")
        plot_labels(tissue_tag_annotation, alpha=0.7)
        print("[INFO] Visualization complete.")

    print("[INFO] Classification finished successfully.")

    return tissue_tag_annotation if copy else None


def overlay_labels(im1, im2, alpha=0.8, show=True):
    """
    Helper function to merge 2 images.

    Parameters
    ----------
    im1 : np.array
        1st image.
    im2 : np.array
        2nd image.
    alpha : float, optional
        Blending factor, by default 0.8.
    show : bool, optional
        If to show the merged plot or not, by default True.

    Returns
    -------
    None | numpy.ndarray
        The merged image array if show is False, otherwise None.
    """

    # Generate overlay image
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.dpi"] = 100
    out_img = np.zeros(im1.shape, dtype=im1.dtype)
    out_img[:, :, :] = (alpha * im1[:, :, :]) + ((1 - alpha) * im2[:, :, :])
    out_img[:, :, 3] = 255
    if show:
        plt.imshow(out_img, origin='lower')

    return None if show else out_img


def plot_labels(tissue_tag_annotation, alpha=0.8):
    """
    Helper function to plot the annotation labels on the image.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image, label_image and annotation map.
    alpha : float, optional
        Blending factor, by default 0.8.

    Returns
    -------
    None
    """

    annotation = rgb_from_labels(tissue_tag_annotation)
    # Read-only preview plot: materialise a local numpy view (see rgb_from_labels) without
    # mutating a file-backed tissue_tag_annotation.
    return overlay_labels(np.asarray(tissue_tag_annotation.image), annotation, alpha, show=True)


def segmenter(tissue_tag_annotation, plot_size=1024, invert_y=False, use_datashader=False,
              annotation_prefix="object", label_aggregator='max', file_backed=False, work_dir=None):
    """
    Interactive annotation tool to segment image using Panel to switch between morphology and annotation.

    As with ``annotator``, the annotation layer is now rendered natively from ``label_image``.
    Because the colour mapping is a plot option rather than baked pixel data, the palette is
    fixed up-front to cover the full uint8 label range. Adding an object therefore only requires
    pushing the updated label_image through the Pipe: the colour options are never touched, and
    the bokeh plot (and the user's zoom/pan) is never rebuilt.

    Object colours are now assigned deterministically by cycling ``SEGMENTER_COLORPOOL`` rather
    than being drawn at random, so neighbouring objects still contrast but the result is
    reproducible.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image and annotation map.
    plot_size: int, default=1024
        Figure size for plotting.
    invert_y : bool
        invert plot along y axis
    use_datashader : bool, optional
        If we should use datashader for rendering the image. Recommended for high resolution image. Default is False.
    annotation_prefix : str, optional
        Prefix for annotations. Default is "object".
    label_aggregator : str, optional
        Reduction used to downsample the label image when zoomed out. See ``label_image_overlay``.
        Default is 'max'.
    file_backed : bool, optional
        Keep ``image``/``label_image`` on disk (Zarr, via ``tissue_tag.file_backed``) rather than
        fully in RAM. See ``annotator`` for details; the same bounding-box-scoped write/undo
        applies here to both drawn objects and eraser strokes. Default is False.
    work_dir : str, optional
        Directory to hold the on-disk Zarr stores when ``file_backed`` triggers a fresh conversion.
        Defaults to a new temporary directory. Ignored if ``tissue_tag_annotation`` is already
        file-backed.

    Returns
    -------
    Panel app
        A panel application object with segmenter tool.
    """

    label_was_missing = tissue_tag_annotation.label_image is None and not tissue_tag_annotation.file_backed
    use_file_backed = file_backed or tissue_tag_annotation.file_backed
    if use_file_backed:
        if not tissue_tag_annotation.file_backed:
            tissue_tag_annotation.to_file_backed(work_dir or tempfile.mkdtemp())
        else:
            # to_file_backed() already calls this; also cover objects that were made
            # file-backed by direct field assignment rather than via that method.
            from tissue_tag import file_backed as fb
            fb.configure_dask_for_low_ram()
        if label_was_missing:
            tissue_tag_annotation.annotation_map = OrderedDict({})
    elif label_was_missing:
        tissue_tag_annotation.label_image = np.zeros(
            (tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]),
            dtype=DEFAULT_LABEL_DTYPE
        )
        tissue_tag_annotation.annotation_map = OrderedDict({})

    if tissue_tag_annotation.annotation_map is None:
        tissue_tag_annotation.annotation_map = OrderedDict({})

    max_labels = np.iinfo(tissue_tag_annotation.label_image.dtype).max

    palette_colors = (SEGMENTER_COLORPOOL[i % len(SEGMENTER_COLORPOOL)] for i in range(max_labels))
    palette = hex_palette_generator(OrderedDict(enumerate(palette_colors)))

    update_button = pn.widgets.Button(name='Update', button_type='primary')
    revert_button = pn.widgets.Button(name='Revert', button_type='danger', disabled=True)
    label_image_opacity = pn.widgets.FloatSlider(name='Label overlay', value=0.5, start=0, end=1, step=0.1)

    label_pipe = Pipe(data=tissue_tag_annotation.label_image)

    ds_img = base_image_element(tissue_tag_annotation.image, plot_size=plot_size,
                                invert_y=invert_y, use_datashader=use_datashader)
    ds_anno = label_image_overlay(label_pipe, palette, plot_size=plot_size, invert_y=invert_y,
                                  use_datashader=use_datashader, annotation_aggregator=label_aggregator,
                                  opacity_param_object=label_image_opacity)

    plot_list = [ds_img, ds_anno]

    path_object = hv.Path([]).opts(line_width=3, line_alpha=0.7)
    draw_object = hv.streams.PolyDraw(source=path_object, show_vertices=True, num_objects=300, drag=True)
    edit_object = hv.streams.PolyEdit(source=path_object, vertex_style={'color': 'red'}, shared=True)
    plot_list.append(path_object)

    erase_path_object = hv.Path([]).opts(line_width=3, line_alpha=1, line_color="black")
    erase_object = CustomPolyDraw(source=erase_path_object, num_objects=300, show_vertices=True, drag=True,
                                  tooltip="Eraser", vertex_style={'color': 'black'})
    edit_erase_object = hv.streams.PolyEdit(source=erase_path_object, shared=True)
    plot_list.append(erase_path_object)

    tab_object = pn.panel(hv.Overlay(plot_list).collate())
    p = pn.Column(pn.Row(label_image_opacity, update_button, revert_button), tab_object)

    previous_label = None if use_file_backed else tissue_tag_annotation.label_image.copy()
    previous_annotation_map = tissue_tag_annotation.annotation_map.copy()
    previous_blocks = []  # file-backed only: (y0, y1, x0, x1, previous_block) from the last Update

    def update_segmenter(event):
        nonlocal previous_label, previous_annotation_map, previous_blocks

        if not event:
            return

        update_button.disabled = True

        previous_annotation_map = tissue_tag_annotation.annotation_map.copy()

        existing_object_count = len(tissue_tag_annotation.annotation_map.keys()) + 1

        new_object_strokes = []
        if draw_object.data['xs']:
            for o in range(len(draw_object.data['xs'])):
                label_value = existing_object_count + o
                if label_value > max_labels:
                    warnings.warn(
                        f"Reached the {max_labels}-object limit for label_image; "
                        "further objects were ignored."
                    )
                    break

                new_object_strokes.append((draw_object.data['xs'][o], draw_object.data['ys'][o], label_value))

                # Record the colour this object was actually drawn in, taken from the fixed
                # palette rather than picked at random, so annotation_map and the rendered
                # overlay can never disagree.
                tissue_tag_annotation.annotation_map[
                    f"{annotation_prefix}_{label_value}"
                ] = palette[label_value - 1]

        if use_file_backed:
            # Erase strokes first, then new objects -- same order the in-memory path applies them.
            erase_strokes = [
                (erase_object.data['xs'][o], erase_object.data['ys'][o], 0)
                for o in range(len(erase_object.data['xs']))
            ] if erase_object.data['xs'] else []
            writer = tissue_tag_annotation.label_writer()
            previous_blocks = _write_polygon_strokes_file_backed(writer, erase_strokes + new_object_strokes)
            tissue_tag_annotation.refresh_label_view()
            label_pipe.send(tissue_tag_annotation.label_image)
        else:
            previous_label = tissue_tag_annotation.label_image.copy()
            updated_labels = tissue_tag_annotation.label_image.copy()

            if erase_object.data['xs']:
                for o in range(len(erase_object.data['xs'])):
                    x = np.array(erase_object.data['xs'][o]).astype(int)
                    y = np.array(erase_object.data['ys'][o]).astype(int)
                    rr, cc = polygon(y, x)
                    inshape = (updated_labels.shape[0] > rr) & (0 < rr) & \
                              (updated_labels.shape[1] > cc) & (0 < cc)
                    updated_labels[rr[inshape], cc[inshape]] = 0

            for xs, ys, label_value in new_object_strokes:
                x = np.array(xs).astype(int)
                y = np.array(ys).astype(int)
                rr, cc = polygon(y, x)
                inshape = (updated_labels.shape[0] > rr) & (0 < rr) & \
                          (updated_labels.shape[1] > cc) & (0 < cc)
                updated_labels[rr[inshape], cc[inshape]] = label_value

            tissue_tag_annotation.label_image = updated_labels
            label_pipe.send(updated_labels)

        clear_draw_stream(draw_object)
        clear_draw_stream(erase_object)

        revert_button.disabled = False
        update_button.disabled = False

    def revert_segmenter(event):
        if not event:
            return

        update_button.disabled = True

        tissue_tag_annotation.annotation_map = previous_annotation_map.copy()
        if use_file_backed:
            writer = tissue_tag_annotation.label_writer()
            _revert_polygon_strokes_file_backed(writer, previous_blocks)
            tissue_tag_annotation.refresh_label_view()
        else:
            tissue_tag_annotation.label_image = previous_label.copy()
        label_pipe.send(tissue_tag_annotation.label_image)

        clear_draw_stream(draw_object)
        clear_draw_stream(erase_object)

        revert_button.disabled = True
        update_button.disabled = False

    pn.bind(update_segmenter, update_button, watch=True)
    pn.bind(revert_segmenter, revert_button, watch=True)

    return p


def _write_disks_batched_file_backed(writer, points, r, value, preserve_existing=False):
    """
    Write same-radius, same-value disks centred at each ``(cy, cx)`` in ``points`` directly onto
    ``writer``'s on-disk label store, in place of ``skimage.draw.disk(...)`` + a full-array
    assignment. Used by ``gene_labels_from_adata``'s file-backed path for both the sparse
    background grid and gene-marked cell positions.

    Writes are batched by on-disk chunk (``writer.chunk_shape``) rather than done one bounding
    box per point: for a large sparse grid (thousands-tens of thousands of points, as
    ``space_every_spots`` produces on a big image), one read+write round trip per point does not
    scale -- at that call volume it was observed to make zarr's sync/async bridge spin up an
    unbounded number of OS threads rather than merely being slow. Grouping points by which
    on-disk chunk they fall in (padding each group's read/write region by ``r`` to catch a disk
    spilling slightly past its chunk's edge) bounds the number of round trips by the number of
    *chunks* actually touched, independent of how many points land in each one.

    Parameters
    ----------
    writer: file_backed.WritableLabelStore
    points: iterable of (cy, cx)
        Disk centres, in array (row, column) coordinates.
    r: float
        Disk radius.
    value: int
        Label value to write.
    preserve_existing: bool, optional
        See ``file_backed.WritableLabelStore.write_masked``. Default False.
    """

    points = list(points)
    if not points:
        return

    chunk_h, chunk_w = writer.chunk_shape
    shape = writer.shape
    pad = int(np.ceil(r))

    buckets = {}
    for cy, cx in points:
        key = (int(cy) // chunk_h, int(cx) // chunk_w)
        buckets.setdefault(key, []).append((cy, cx))

    for (cy_idx, cx_idx), pts in buckets.items():
        base_y0, base_x0 = cy_idx * chunk_h, cx_idx * chunk_w
        y0 = max(0, base_y0 - pad)
        y1 = min(shape[0], base_y0 + chunk_h + pad)
        x0 = max(0, base_x0 - pad)
        x1 = min(shape[1], base_x0 + chunk_w + pad)
        if y0 >= y1 or x0 >= x1:
            continue

        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for cy, cx in pts:
            rr, cc = disk((cy - y0, cx - x0), r, shape=(y1 - y0, x1 - x0))
            local_mask[rr, cc] = True

        writer.write_masked(y0, y1, x0, x1, local_mask, value, preserve_existing=preserve_existing)


def _background_labels_intensity_file_backed(writer, image, r, intensity_threshold, grid_unit_size,
                                              label, preserve_existing):
    """
    File-backed counterpart of ``background_labels_intensity``: paints a disk of ``label`` at
    every sparse grid point whose pixel is bright enough to count as background, writing straight
    to ``writer``'s on-disk label store instead of building a full-resolution numpy array.

    The grid itself (``square_grid``) is already sparse (spaced ``grid_unit_size`` spot-diameters
    apart) and cheap regardless of image size, but a large image can still produce tens of
    thousands of grid points. Testing each point's brightness is batched into a single
    ``dask.array.vindex`` gather (paired/vectorized indexing, like
    ``organaxis.get_annotations_for_objects``), so only the on-disk chunks actually containing a
    grid point are ever touched; the resulting disk writes are batched by chunk too (see
    ``_write_disks_batched_file_backed``) rather than done one at a time.

    Parameters
    ----------
    writer: file_backed.WritableLabelStore
    image: xarray.DataArray
        Dask-backed RGB(A) image (``tissue_tag_annotation.image``).
    r: float
        Disk radius for each background spot.
    intensity_threshold: int
    grid_unit_size: int
    label: int
        Label value for background spots.
    preserve_existing: bool
        See ``file_backed.WritableLabelStore.write_masked``.
    """

    shape = writer.shape
    grid = square_grid(r, shape, grid_unit_size).T
    if grid.size == 0:
        return

    ys = grid[:, 1].astype(int)
    xs = grid[:, 0].astype(int)
    valid = (ys >= 0) & (xs >= 0) & (ys < shape[0]) & (xs < shape[1])
    ys, xs = ys[valid], xs[valid]
    if ys.size == 0:
        return

    n_bands = image.sizes['band']
    points = image.data.vindex[ys, xs, :].compute().astype(np.float64)

    if n_bands == 4:
        grayscale_vals = points[:, :3] @ [0.2989, 0.5870, 0.1140]
    elif n_bands == 3:
        grayscale_vals = points @ [0.2989, 0.5870, 0.1140]
    else:
        raise ValueError("Unexpected number of channels in imarray.")

    is_background = grayscale_vals > intensity_threshold
    background_points = list(zip(ys[is_background].tolist(), xs[is_background].tolist()))
    _write_disks_batched_file_backed(writer, background_points, r, label, preserve_existing=preserve_existing)


def gene_labels_from_adata(adata, gene_markers, tissue_tag_annotation, diameter, override_labels=False,
                           space_every_spots=10, normalize=True, unassigned_colour="yellow", intensity_threshold=230,
                           copy=False):
    """
    Assign labels to training spots based on gene expression from an existing AnnData object.

    Parameters
    ----------
    adata : AnnData
        Pre-loaded AnnData object containing gene expression data.
    gene_markers : dict
        Dictionary mapping annotation to pairs of marker genes and number of spots to assign.
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with image and annotation map.
    diameter : float
        Radius of the spots.
    override_labels : bool
        Remove existing label_image and replace with new labels.
    space_every_spots : int, optional
        Spacing between background spots. Default is 10.
    normalize : bool
        if to normalize gene expression by default parametres calculated by - scanpy.pp.normalize_total()
    unassigned_colour : str, optional
        Color for unassigned labels. Default is "yellow".
    intensity_threshold: int, optional
        Threshold above which pixels are considered background. Default is 230.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the label_image in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with updated label_image based on the gene expression if copy is True,
        otherwise None.
    """

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation
    use_file_backed = tissue_tag_annotation.file_backed

    if use_file_backed:
        # A file-backed label_image is never None (to_file_backed() always leaves a store in
        # place, all-zero if none existed), so recreating it here (override_labels) or leaving
        # it as the writable target (add-on-top) covers what the in-memory branch below does.
        from tissue_tag import file_backed as fb
        fb.configure_dask_for_low_ram()
        if override_labels:
            print("Label image is not empty. Will replace with an empty label_image.")
            fb.zeros_zarr(
                (int(tissue_tag_annotation.image.sizes['y']), int(tissue_tag_annotation.image.sizes['x'])),
                tissue_tag_annotation.label_store, overwrite=True,
            )
            tissue_tag_annotation.refresh_label_view()
        else:
            print("Will add new gene labels on top of old label_image.")
    elif tissue_tag_annotation.label_image is not None:
        print("Label image is not empty.")
        if override_labels:
            # Initialize label image
            print("Will replace with an empty label_image.")
            tissue_tag_annotation.label_image = np.zeros(
                (tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]), dtype=np.uint8)
        else:
            print("Will add new gene labels on top of old label_image.")
    else:  # if the label_image spot is empty then create a blank one
        tissue_tag_annotation.label_image = np.zeros(
            (tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]), dtype=np.uint8)

    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide an annotation map.")
    else:
        tissue_tag_annotation.annotation_map = OrderedDict(tissue_tag_annotation.annotation_map)
        tissue_tag_annotation.annotation_map["unassigned"] = unassigned_colour
        tissue_tag_annotation.annotation_map.move_to_end("unassigned", last=False)

    # Filter adata to match df indices
    adata = adata[tissue_tag_annotation.positions.index.intersection(adata.obs.index)]
    r = diameter / 2 * tissue_tag_annotation.ppm

    # Extract coordinates
    if use_file_backed:
        # Writes go straight to the on-disk label store, batched by on-disk chunk -- see
        # _background_labels_intensity_file_backed and _write_disks_batched_file_backed.
        # Background spots use preserve_existing=True so they never clobber pixels the user (or
        # an earlier call) already annotated, matching the in-memory branch's
        # `labels[mask] = tissue_tag_annotation.label_image[mask]` merge below.
        writer = tissue_tag_annotation.label_writer()
        _background_labels_intensity_file_backed(
            writer, tissue_tag_annotation.image, r=r,
            intensity_threshold=intensity_threshold, grid_unit_size=space_every_spots,
            label=1, preserve_existing=True,
        )
    else:
        labels = background_labels_intensity(tissue_tag_annotation.label_image.shape[:2],
                                             imarray=tissue_tag_annotation.image, r=r,
                                             intensity_threshold=intensity_threshold, grid_unit_size=space_every_spots,
                                             label=1)
        mask = tissue_tag_annotation.label_image > 0
        labels[mask] = tissue_tag_annotation.label_image[mask]  # add old labels if these are not empty

    if normalize:
        normalize_total(adata)

    # Assign labels based on gene expression
    for label, gene_list in gene_markers.items():
        # Get the expected color for the marker
        label_color = tissue_tag_annotation.annotation_map.get(label, "N/A")
        print(f"🧬 Processing label: '{label}' | Color: {label_color} | Genes: {[gene for gene, _ in gene_list]}")

        combined_gene_indices = []

        for gene, top_n in gene_list:
            GeneIndex = np.where(adata.var_names.str.fullmatch(gene))[0]
            if GeneIndex.size == 0:
                print(f"Warning: Gene {gene} not found in AnnData. Skipping.")
                continue

            GeneData = adata.X[:, GeneIndex].todense().A1  # Flatten to 1D array
            nonzero_indices = np.where(GeneData > 0)[0]

            if len(nonzero_indices) == 0:
                print(f"Warning: No non-zero expression for gene {gene}. Skipping.")
                continue

            # Build a DataFrame to sort and shuffle
            gene_df = pd.DataFrame({
                "barcode": adata.obs.index[nonzero_indices],
                "expression": GeneData[nonzero_indices]
            })

            # Shuffle within expression levels to avoid spatial artifacts: shuffle all rows once,
            # then stable-sort descending by expression so ties keep their shuffled relative
            # order. (Equivalent to, but avoids, groupby("expression").apply(sample) -- pandas
            # >=2.2 drops the grouping column from that apply's result, which would otherwise
            # break the sort_values("expression") below with a KeyError.)
            gene_df = gene_df.sample(frac=1)

            # Now sort by expression descending
            gene_df_sorted = gene_df.sort_values("expression", ascending=False, kind="stable")

            # Take top N
            actual_top_n = min(top_n, len(gene_df_sorted))
            selected_barcodes = gene_df_sorted["barcode"].iloc[:actual_top_n]

            combined_gene_indices.extend(selected_barcodes)

        # Remove duplicates and convert to a set for faster lookups later
        combined_gene_indices = set(combined_gene_indices)

        # Assign labels
        for idx, sub in enumerate(tissue_tag_annotation.annotation_map.keys()):
            if sub == label:
                label_value = idx

        # Gene-marked cells always override whatever is there (background or existing
        # annotation), applied last -- matches the in-memory branch's unconditional
        # `labels[disk(...)] = label_value + 1`.
        coords = tissue_tag_annotation.positions.loc[list(combined_gene_indices), ["pxl_row", "pxl_col"]].to_numpy()
        if use_file_backed:
            _write_disks_batched_file_backed(
                writer, [(coor[0], coor[1]) for coor in coords], r, label_value + 1, preserve_existing=False,
            )
        else:
            for coor in coords:
                labels[disk((coor[0], coor[1]), r)] = label_value + 1

    if use_file_backed:
        tissue_tag_annotation.refresh_label_view()
    else:
        tissue_tag_annotation.label_image = labels

    return tissue_tag_annotation if copy else None


def background_labels_intensity(shape, imarray, r, intensity_threshold=230, grid_unit_size=10, label=1):
    """
    Generate background labels based on intensity (bright pixels in brightfield images).

    Parameters
    ----------
    shape : tuple
        Shape of the training labels array.
    imarray : numpy.ndarray
        RGB image used to identify bright background areas.
    r : float
        Radius of the spots.
    intensity_threshold : int, optional
        Threshold above which pixels are considered background. Default is 230.
    grid_unit_size : int, optional
        Spacing between background spots. Default is 10.
    label : int, optional
        Label value for background spots. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array containing the background labels.
    """

    # Convert RGBA to grayscale using only RGB channels
    if imarray.shape[-1] == 4:  # RGBA
        grayscale = np.dot(imarray[..., :3], [0.2989, 0.5870, 0.1140])  # Standard grayscale conversion
    elif imarray.shape[-1] == 3:  # RGB
        grayscale = np.dot(imarray, [0.2989, 0.5870, 0.1140])
    else:
        raise ValueError("Unexpected number of channels in imarray.")

    # Identify bright pixels in the grayscale image (background areas)
    background_mask = grayscale > intensity_threshold

    training_labels = np.zeros(shape, dtype=np.uint8)
    grid = square_grid(r, shape, grid_unit_size).T

    print(imarray.shape)

    for coor in grid:
        y, x = int(coor[1]), int(coor[0])  # Ensure integer indices
        if y >= background_mask.shape[0] or x >= background_mask.shape[1]:  # Avoid out-of-bounds indexing
            continue
        if np.any(background_mask[y, x]):  # Use `.any()` if needed
            training_labels[disk((y, x), r, shape=shape)] = label

    return training_labels


def square_grid(spot_size, shape, grid_unit_size):
    """
    Generate a square grid

    Parameters
    ----------
    spot_size : float
        Size of the spots.
    shape : tuple
        Shape of the grid (height, width).
    grid_unit_size : int
        Spacing between background spots.

    Returns
    -------
    numpy.ndarray
        Array containing the coordinates of the grid.
    """
    # Define step sizes
    dx = spot_size * grid_unit_size  # Horizontal spacing
    dy = spot_size * grid_unit_size  # Vertical spacing

    # Generate meshgrid for a square grid
    x_coords = np.arange(spot_size, shape[0] - spot_size, dx)
    y_coords = np.arange(spot_size, shape[1] - spot_size, dy)

    gx, gy = np.meshgrid(x_coords, y_coords)

    # Stack the x and y coordinates
    positions = np.vstack([gy.ravel(), gx.ravel()])

    return positions


def median_filter(tissue_tag_annotation, filter_radius=10, downsampling_factor=1, copy=False):
    """
    Apply a median filter to the label image of the TissueTagAnnotation object.

    Parameters
    ----------
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with label_image.
    filter_radius : int, optional
        Radius of the median filter. Default is 10.
    downsampling_factor: integer, optional
        Downsampling factor to resize label_image prior to running the median filter. Default value of 1 means no downsampling is performed.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the label_image in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with filtered label_image if copy is True,otherwise None.
    """

    from skimage.filters import median
    from skimage.morphology import disk

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation
    tissue_tag_annotation = _ensure_in_memory(tissue_tag_annotation)
    label_image_shape = tissue_tag_annotation.label_image.shape
    r = int(filter_radius * tissue_tag_annotation.ppm)

    if downsampling_factor > 1:
        r = r // downsampling_factor
        tissue_tag_annotation.label_image = downsample_segmentation(tissue_tag_annotation.label_image,
                                                                    (downsampling_factor, downsampling_factor))[0]

    tissue_tag_annotation.label_image = median(tissue_tag_annotation.label_image, footprint=disk(r))

    if downsampling_factor > 1:
        tissue_tag_annotation.label_image = cv2.resize(tissue_tag_annotation.label_image, label_image_shape[::-1],
                                          interpolation=cv2.INTER_NEAREST)

    return tissue_tag_annotation if copy else None


def assign_annotation_label_to_positions(tissue_tag_annotation, annotation_column='annotation', copy=False):
    """
    Assign annotation to each cell in positions data frame based on the cell centroid co-ordinates.

    Parameters
    ----------
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with label_image and positions data frame.
    annotation_column : str, optional
        Column name for the annotation values. Default is 'annotation'.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the positions data frame in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with updated positions data frame if copy is True,otherwise None.
    """
    if tissue_tag_annotation.label_image is None:
        raise ValueError("Label image is missing. Please annotate the image first.")

    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide an annotation map.")

    if tissue_tag_annotation.positions is None:
        raise ValueError("Positions data frame is missing. Please provide positions data frame.")

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation
    # get_annotations_for_objects() does its own paired (non-outer-product) point indexing for
    # a file-backed label_image, touching only the chunks containing the requested points -- see
    # its docstring/comments in organaxis.py. No materialisation needed here.

    coord_df = tissue_tag_annotation.positions[["pxl_row", "pxl_col"]].rename(columns={"pxl_row":"x", "pxl_col":"y"})
    tissue_tag_annotation.positions[annotation_column] = get_annotations_for_objects(tissue_tag_annotation, coord_df)

    return tissue_tag_annotation if copy else None


def plot_cell_label_annotations(tissue_tag_annotation, cell_diameter=5.0, annotation_column='annotation', alpha=0.8):
    """
    Helper function to plot the cell annotation labels on the image.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image, label_image and annotation map.
    cell_diameter: float, optional
        Desired marker diameter in microns. Defaults to 5.0.
        Recommended values are 55 for Visium and 2, 8 or 16 depending on bin size selected for Visium HD.
    annotation_column : str, optional
        Column name for the annotation values. Default is 'annotation'.
    alpha : float, optional
        Blending factor, by default 0.8.

    Returns
    -------
    None
    """

    if tissue_tag_annotation.positions is None:
        raise ValueError("Positions data frame is missing. Please provide positions data frame.")
    if annotation_column not in tissue_tag_annotation.positions.columns is None:
        raise ValueError("Annotation column in posititions data frame is missing. Please assign annotation to cells first.")

    fig, ax = plt.subplots(figsize=(10, 10), dpi = 100)

    base_img = np.zeros(tissue_tag_annotation.image.shape, dtype=tissue_tag_annotation.image.dtype)
    base_img[:, :, :] = (alpha * tissue_tag_annotation.image[:, :, :])
    base_img[:, :, 3] = 255
    ax.imshow(base_img, origin='lower')

    marker_size_pixels = cell_diameter * tissue_tag_annotation.ppm
    for ann, color in tissue_tag_annotation.annotation_map.items():
        selected_position = tissue_tag_annotation.positions[tissue_tag_annotation.positions[annotation_column] == ann]
        zipped = np.broadcast(selected_position["pxl_col"], selected_position["pxl_row"], marker_size_pixels * 0.5)
        patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
        collection = PatchCollection(patches)
        collection.set_facecolor(color)
        collection.set_edgecolor(color)
        collection.set_alpha(1-alpha)
        ax.add_collection(collection)

    plt.show()
    plt.close(fig)