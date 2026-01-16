import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import polygon
import skimage.transform
import skimage.draw
import scipy.ndimage
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None


def run_tissuetag_visium_distance_pipeline(
    adata,
    tissue_tag_annotation,
    grid_unit_size,
    knn = 5,
    annotation_column = "annotation",
    max_distance = None,
    drop_unassigned = True,
    plot = True,
    copy = False
):
    """
    Run Visium distance-mapping pipeline which are composed of the following steps:
        1. Grid generation
        2. Annotation distance computation
        3. Mapping of annotation to spot coordinates
        4. Updating obs dataframe with distance features

    Feature columns are named like:
        L2_dist_annotation_white_matter_g15_k10
    or:
        L2_dist__white_matter_g15_k10

    Parameters
    ----------
    adata : anndata
        Anndata object containing data from Visium/Visium HD.
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with label_image and positions data frame.
    grid_unit_size : float
        Distance between grid points in microns (µm).
    knn : int
        Number of nearest neighbors to consider. Default is 5.
    annotation_column : str, optional
        Column name for the annotation values. Default is 'annotation'.
    max_distance : float, optional
        Maximum allowable distance for matching points in microns (µm).
        If None (default), the max distance will be set to 3x grid_unit_size.
    drop_unassigned : bool, optional
        Ignore entries which have no annotation (i.e. unassigned). Default is True.
    plot : bool, optional
        If True, plots the coordinates of the grid space and the spot space to verify alignment. Default is True.
    copy : bool, optional
        Return a new copy of the anndata object instead of modifying it in place. Default is False.

    Returns
    -------
    Anndata
        Anndata object updated with annotation for cells in anndata.obs.
    pandas.DataFrame
        DataFrame containing the grid coordinates and corresponding annotation values.
    pandas.DataFrame
        DataFrame containing cell coordinates and corresponding annotation values.
    dict
        Dictionary object containing a list of overwritten column(s) and newly added column(s) in anndata.obs.
    """
    adata = adata.copy() if copy else adata

    if 'spatial' not in adata.uns:
        raise KeyError("Spatial information is missing from anndata object. "
                       "This pipeline require anndata containing Visium/Visium HD data.")

    # Generate grid (in microns as ppm_out=1 means 1 px/µm)
    grid_df = generate_grid_from_annotation(
        tissue_tag_annotation,
        grid_unit_size=grid_unit_size,
        ppm_out=1,
        annotation_column=annotation_column,
    )

    # Drop entries where annotation is assigned (optional)
    if drop_unassigned and annotation_column in grid_df:
        grid_df = grid_df[grid_df[annotation_column] != "unassigned"]

    # Compute per-annotation distances (micron scale)
    calculate_distance_to_annotations(
        grid_df,
        knn=knn,
        annotation_column=annotation_column,
        copy=False
    )

    # Get resolution of the Visium/Visium HD library
    spatial_meta = adata.uns["spatial"]
    library_id = next(iter(spatial_meta.keys()))
    microns_per_pixel = spatial_meta[library_id]["scalefactors"]["microns_per_pixel"]
    ppm_target = 1.0 / microns_per_pixel  # pixels per micron

    # Map annotation from grid to spots
    target_df = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"])

    if max_distance is None:
        max_distance = 3.0 * grid_unit_size

    mapped_df = map_annotations_to_target(
        df_source=grid_df,
        df_target=target_df,
        ppm_source=1.0,         # grid in microns
        ppm_target=ppm_target,  # Visium pixels per micron
        plot=plot,
        max_distance=max_distance,  # microns
        copy=True
    )

    # Align index to anndata and extract distance columns only
    dist_cols = [c for c in mapped_df.columns if c.startswith("L2_dist_")]
    mapped_df = mapped_df.loc[adata.obs.index, dist_cols]

    # Update column name with grid unit size and nhood size
    suffix = f"_g{int(round(grid_unit_size))}_k{int(knn)}"
    mapped_df.columns = [c + suffix for c in mapped_df.columns]

    # Track which columns are being overwritten vs newly created
    existing = set(dist_cols).intersection(set(adata.obs.columns))
    new = set(dist_cols).difference(set(adata.obs.columns))

    # Overwrite (and create) in one shot
    adata.obs[mapped_df.columns] = mapped_df

    return adata, grid_df, mapped_df, {"overwritten": existing, "added": new}


def assign_annotation_label_to_object(tissue_tag_annotation, coord_df):
    """
    Assign annotation to objects based on their coordinate.

    Parameters
    ----------
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with label_image and positions data frame.
    coord_df : pandas.DataFrame
        DataFrame object containing two columns representing scaled x,y coordinates of objects.

    Returns
    -------
    pandas.Series
        Series object containing annotation for each object in coord_df
    """

    if tissue_tag_annotation.label_image is None:
        raise ValueError("Label image is missing. Please annotate the image first.")

    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide an annotation map.")

    if coord_df.shape[1] != 2:
        raise ValueError("Please provide a DataFrame containing two columns with x,y coordinates only.")

    annotation_label_mapping = {i + 1: v for i, v in enumerate(tissue_tag_annotation.annotation_map.keys())}
    annotation_ids = tissue_tag_annotation.label_image[np.rint(coord_df["x"]).astype(int), np.rint(coord_df["y"]).astype(int)]
    annotation_labels = pd.Series(annotation_ids).map(lambda annotation_id: annotation_label_mapping.get(annotation_id, "Unknown"))

    return annotation_labels.values


def calculate_axis(feature_df, feature_columns, output_column, weights=(0.2, 0.8), copy=False):
    """
    Calculate a unimodal normalized axis based on 2 or 3 ordered features.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Input DataFrame containing featrues to calculate axis.
    feature_columns : list of str
        List of column names containing features to calculate axis.
        If two columns are specified, a simple 2-point axis (S1 -> S2) will be calculated.
        If three columns are specified, a 3-point axis (S1 -> S2 -> S3) will be calculated.
    output_column : str
        Name of the output column to store the calculated axis.
    weights : tuple of float, optional
        Weights for the 3-point axis (w0, w1). Default (0.2,0.8).
        The default value ensure monotonically increasing with distance for structure S3 if this structure is large.

    Returns
    -------
    None | pandas.DataFrame
        DataFrame with the calculated axis column if copy is True, otherwise None.
    """

    feature_df = feature_df.copy() if copy else feature_df

    if not 2 <= len(feature_columns) <= 3:
        raise ValueError("Please specify either 2 or 3 feature columns to calculate 2 or 3 point axis.")

    if len(feature_columns) == 2:
        axis1 = (feature_df[feature_columns[0]] - feature_df[feature_columns[1]]) / (feature_df[feature_columns[0]] + feature_df[feature_columns[1]])
        feature_df[output_column] = axis1

    elif len(feature_columns) == 3:
        if len(weights) != 2:
            raise ValueError("Please provide 2 weights (w0, w1) for 3-point axis.")

        axis1 = (feature_df[feature_columns[0]] - feature_df[feature_columns[1]]) / (feature_df[feature_columns[0]] + feature_df[feature_columns[1]])
        axis2 = (feature_df[feature_columns[1]] - feature_df[feature_columns[2]]) / (feature_df[feature_columns[1]] + feature_df[feature_columns[2]])
        feature_df[output_column] = weights[0] * axis1 + weights[1] * axis2

    return feature_df if copy else None


def generate_hires_grid(im, grid_unit_size, pixels_per_micron):
    """
    Creates a hexagonal grid of a specified size and density.
    
    Parameters
    ----------
    im : numpy.ndarray
        Image to fit the grid on (mostly for dimensions).
    grid_unit_size : float
        Distance between spots in grid.
    pixels_per_micron : float
        The resolution of the image in pixels per micron.

    Returns
    -------
    numpy.ndarray
        Hexagonal grid coordinates
    """

    # Step size in pixels for spot_to_spot microns
    step_size_in_pixels = grid_unit_size * pixels_per_micron

    # Generate X-axis and Y-axis grid points
    X1 = np.arange(step_size_in_pixels, im.shape[1] - 2 * step_size_in_pixels, step_size_in_pixels * np.sqrt(3) / 2)
    Y1 = np.arange(step_size_in_pixels, im.shape[0] - step_size_in_pixels, step_size_in_pixels)

    # Shift every other column by half a step size (for staggered pattern in columns)
    positions = []
    for i, x in enumerate(X1):
        if i % 2 == 0:  # Even columns (no shift)
            Y_shifted = Y1
        else:  # Odd columns (shifted by half)
            Y_shifted = Y1 + step_size_in_pixels / 2

        # Combine X and Y positions, and check for boundary conditions
        for y in Y_shifted:
            if 0 <= x < im.shape[1] and 0 <= y < im.shape[0]:
                positions.append([x, y])

    return np.array(positions).T


def create_disk_kernel(radius, shape):
    """
    Create a disk-shaped kernel for filtering.

    Parameters
    ----------
    radius : int
        Radius of the disk.
    shape : tuple
        Shape of the kernel (height, width).

    Returns
    -------
    numpy.ndarray
        Disk-shaped kernel.
    """

    rr, cc = skimage.draw.disk((radius, radius), radius, shape=shape)
    kernel = np.zeros(shape, dtype=bool)
    kernel[rr, cc] = True
    return kernel


def generate_grid_from_annotation(tissue_tag_annotation, grid_unit_size, ppm_out=1, annotation_column='annotation'):
    """
    Generate a grid and assign annotation values to each grid point based on the median value of the annotation image.

    Parameters
    ----------
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object containing label_image
    grid_unit_size : float
        Distance between spots in grid.
    ppm_out : float
        The resolution of the output grid in pixels per micron.
    annotation_column : str, optional
        Column name for the annotation values. Default is 'annotation'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the grid coordinates and corresponding annotation values.
    """

    print(f'Generating grid with spacing - {grid_unit_size}, '
          f'from annotation resolution of - {tissue_tag_annotation.ppm} ppm')

    positions = generate_hires_grid(tissue_tag_annotation.label_image, grid_unit_size,
                                    tissue_tag_annotation.ppm).T  # Transpose for correct orientation

    radius = int(round((grid_unit_size / 2) * tissue_tag_annotation.ppm) - 1)
    kernel = create_disk_kernel(radius, (2 * radius + 1, 2 * radius + 1))

    df = pd.DataFrame(positions, columns=['x', 'y'])
    df['index'] = df.index

    anno_orig = skimage.transform.resize(tissue_tag_annotation.label_image, tissue_tag_annotation.label_image.shape[:2],
                                         preserve_range=True).astype('uint8')
    filtered_image = scipy.ndimage.median_filter(anno_orig, footprint=kernel)

    median_values = [filtered_image[int(point[1]), int(point[0])] for point in positions]
    annotation_label_list = {i + 1: v for i, v in enumerate(tissue_tag_annotation.annotation_map.keys())}
    anno_dict = {idx: annotation_label_list.get(val, "Unknown") for idx, val in enumerate(median_values)}
    number_dict = {idx: val for idx, val in enumerate(median_values)}

    df[annotation_column] = list(anno_dict.values())
    df[annotation_column + '_number'] = list(number_dict.values())

    df['x'] = df['x'] * ppm_out / tissue_tag_annotation.ppm
    df['y'] = df['y'] * ppm_out / tissue_tag_annotation.ppm
    df.set_index('index', inplace=True)

    return df


def map_annotations_to_target(df_source, df_target, ppm_target, ppm_source=1, plot=True, max_distance=50.0, copy=False):
    """
    Map annotations from a source to a target DataFrame based on nearest neighbor matching within a maximum distance.

    Parameters
    ----------
    df_source : pandas.DataFrame
        DataFrame with grid data and annotations.
    df_target : pandas.DataFrame
        DataFrame with target data.
    ppm_source : float
        Pixels per micron of source data.
    ppm_target : float
        Pixels per micron of target data.
    plot : bool, optional
        If True, plots the coordinates of the grid space and the spot space to verify alignment. Default is True.
    max_distance : float
        Maximum allowable distance for matching points. Final max_distance used will be max_distance * ppm_target.
    copy : bool, optional
        Return a new copy of the df_target with the columns from df_source instead of modifying it in place.
        Default is False.

    Returns
    -------
    None | pandas.DataFrame
        DataFrame with additional annotations from the source dataframe if copy is True, otherwise None.
    """

    # Adjust coordinate scaling
    a = np.vstack([df_source['x'] / ppm_source, df_source['y'] / ppm_source]).T
    b = np.vstack([df_target['x'] / ppm_target, df_target['y'] / ppm_target]).T

    # Plot the coordinate spaces if requested, overlaying them in a single plot with different colors and a legend
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.scatter(a[:, 0], a[:, 1], s=5, color='blue', label='Source Space', alpha=0.5)
        plt.scatter(b[:, 0], b[:, 1], s=5, color='orange', label='Target Space', alpha=0.5)
        plt.title('Source and Target Space Coordinates')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()

    # Find nearest neighbors and distances only once
    tree = cKDTree(a)
    distances, indices = tree.query(b, distance_upper_bound=max_distance * ppm_target)

    # Filter valid indices based on distance and within-bounds check
    valid_mask = (indices < len(df_source)) & (distances < max_distance * ppm_target)

    df_target = df_target.copy() if copy else df_target

    # For each annotation, assign the value from the nearest neighbor in the source data
    annotations = df_source.columns.difference(['x', 'y'])
    for k in annotations:
        # Initialize with NaN or None where indices are out of bounds
        if pd.api.types.is_numeric_dtype(df_source[k]):
            df_target[k] = np.nan
        else:
            df_target[k] = None

        # Assign values where distance criteria are met and indices are valid
        valid_indices = indices[valid_mask]
        df_target.loc[valid_mask, k] = df_source.iloc[valid_indices][k].values

    return df_target


def calculate_distance_to_annotations(grid_df, knn=5, logscale=False, annotation_column='annotation', copy=False):
    """
    Calculate the nearest distance for each grid points to all annotation categories..

    Parameters
    ----------
    grid_df : pandas.DataFrame
        DataFrame containing the grid coordinates and annotations.
    knn : int, optional
        Number of nearest neighbors to consider. Default is 5.
    logscale : bool, optional
        Use logarithmic scale (base 10) for distances. Default is False.
    annotation_column : str, optional
        Column name for the annotation values within the grid dataframe. Default is 'annotation'.
    copy : bool, optional
        Return a new copy of the grid with the distance instead of modifying it in place. Default is False.

    Returns
    -------
    None | pandas.DataFrame
        DataFrame containing the grid coordinates and distances to each annotation category if copy is True,
        otherwise None.
    """

    grid_df = grid_df.copy() if copy else grid_df
    print('calculating distance matrix')

    points = np.vstack([grid_df['x'], grid_df['y']]).T
    categories = np.unique(grid_df[annotation_column])

    dist_to_annotations = {c: np.zeros(grid_df.shape[0]) for c in categories}

    for idx, c in enumerate(categories):
        indextmp = grid_df[annotation_column] == c
        if np.sum(indextmp) > knn:
            print(c)
            cluster_points = points[indextmp]
            tree = cKDTree(cluster_points)
            # Get KNN nearest neighbors for each point
            distances, _ = tree.query(points, k=knn)
            # Store the mean distance for each point to the current category
            if knn == 1:
                dist_to_annotations[c] = distances  # No need to take mean if only one neighbor
            else:
                dist_to_annotations[c] = np.mean(distances, axis=1)

    for c in categories:
        if logscale:
            grid_df["L2_dist_log10_" + annotation_column + '_' + c] = np.log10(dist_to_annotations[c])
        else:
            grid_df["L2_dist_" + annotation_column + '_' + c] = dist_to_annotations[c]

    print(dist_to_annotations)

    return grid_df if copy else None


# REMOVE OR UPDATE
def bin_axis(axis_df, axis_column, bin_labels, cutoff_values, copy=False):
    """
    Bins a column of a DataFrame based on cutoff values and assigns manual bin labels.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing axis column.
    axis_anno_name : str
        The name of the column containing axis to be binned.
    ct_order : list of str
        The order of manual bin labels.
    cutoff_values : list of float
        The cutoff values used for binning.
    copy : bool, optional
        Return a new copy of the grid with the distance instead of modifying it in place. Default is False.

    Returns
    -------
    None | pandas.DataFrame
        DataFrame with binned axis if copy is True, otherwise None.
    """

    if len(bin_labels) != (len(cutoff_values) + 1):
        raise ValueError("The number of bin labels and cutoff values are not compatible.")

    axis_df = axis_df.copy() if copy else axis_df

    # Initialize binned column with 'unassigned'
    binned_col = f'binned_{axis_column}'
    axis_df[binned_col] = 'unassigned'

    # Assign bins based on cutoff values
    axis_df.loc[axis_df[axis_column] < cutoff_values[0], binned_col] = bin_labels[0]
    print(f"{bin_labels[0]} = ({axis_column} < {cutoff_values[0]})")

    for idx in range(len(cutoff_values) - 1):
        lower, upper = cutoff_values[idx], cutoff_values[idx + 1]
        axis_df.loc[(axis_df[axis_column] >= lower) & (axis_df[axis_column] < upper), binned_col] = bin_labels[idx + 1]
        print(f"{bin_labels[idx + 1]} = ({axis_column} >= {lower}) & ({axis_column} < {upper})")

    axis_df.loc[axis_df[axis_column] >= cutoff_values[-1], binned_col] = bin_labels[-1]
    print(f"{bin_labels[-1]} = ({axis_column} >= {cutoff_values[-1]})")

    return axis_df if copy else None


def plot_cont(df, x_col, y_col, color_col, cmap='jet', title='L2_distance_plot', s=1, dpi=100, figsize=(10, 10)):
    """
    Plot a scatter plot with color mapping based on a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    color_col : str, optional
        Column name for color mapping.
    cmap : str, optional
        Colormap to use. Default is 'jet'.
    title : str, optional
        Title of the plot. Default is 'L2_distance_plot'.
    s : int, optional
        Size of the points in the scatter plot. Default is 1.
    dpi : int, optional
        Dots per inch for the figure. Default is 100.
    figsize : tuple of int, optional
        Size of the figure in inches. Default is (10, 10).

    Returns
    -------
    None
    """

    plt.figure(dpi=dpi, figsize=figsize)

    # Create an axes instance for the scatter plot
    ax = plt.subplot(111)

    # Create the scatterplot
    scatter = sns.scatterplot(x=x_col, y=y_col, data=df,
                              c=df[color_col], cmap=cmap, s=s,
                              legend=False, ax=ax)  # Use the created axes

    plt.grid(False)
    plt.axis('equal')
    plt.title(title)
    for pos in ['right', 'top', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)

    # Add colorbar
    norm = plt.Normalize(df[color_col].min(), df[color_col].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=title, aspect=30)  # Use the created axes for the colorbar
    cbar.ax.set_position([0.85, 0.25, 0.05, 0.5])  # adjust the position as needed

    plt.show()
