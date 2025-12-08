# calculate the
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tissue_tag.io 
from tissue_tag.io import TissueTagAnnotation
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np



# function to filter some noise 
def median_filter(
        tissue_tag_object, 
        filter_radius=10,  # in microns      
        ):
    from skimage.filters import median 
    from skimage.morphology import disk
    r = int(filter_radius*tissue_tag_object.ppm)
    tissue_tag_object.label_image = median(tissue_tag_object.label_image, footprint=disk(r))




def read_visium_hd(
    spaceranger_dir_path,
    spaceranger_spatial_path,
    bin_resolution  = '16',
    use_resolution = "hires",
    ppm_out = None,
    mapped_image_path = None,
    in_tissue = True,
    plot = False,
):
    """
    Reads 10X Visium HD dataset, including spatial image and metadata.

    Parameters
    ----------
    spaceranger_dir_path : str
        Directory containing Visium HD library data.
    spaceranger_spatial_path : str or Path
        Directory containing the spatial images.
    bin_resolution : str, optional
        Resolution of the Visium HD dataset to use. binnig level can be - '02','08','16'
    use_resolution : str, optional
        Desired image resolution ("mapped_res", "hires", "lowres"). selecting "hires" or "lowres" would use the spaceranger output image
    ppm_out : float, optional
        Target resolution in pixels per micron.
    mapped_image_path : str, optional
        Path to the original full-resolution image that use an input to spaceranger (required if use_resolution is "mapped_res").
    in_tissue : bool, optional
        Whether to include only tissue bins (default: True).
    plot : bool, optional
        Whether to plot the output image (default: False).


    Returns
    -------
    tt annotation object 
    """


    # Load scale factors
    scalefactors_file = spaceranger_dir_path + f'/binned_outputs/square_0{bin_resolution}um/spatial/scalefactors_json.json'
    with open(scalefactors_file, "r") as f:
        scalefactors = json.load(f)

    fullres_ppm = 1/scalefactors["microns_per_pixel"] # get to micron scale from pixels

    # Load tissue positions future proofing
    spaceranger_dir_path = Path(spaceranger_dir_path + f'/binned_outputs/square_0{bin_resolution}um/spatial/')  # Convert to Path
    tissue_positions_file = next(
        (f for f in [
            spaceranger_dir_path / "tissue_positions.parquet",
            spaceranger_dir_path / "tissue_positions.csv",
            spaceranger_dir_path / "tissue_positions_list.csv"
        ] if f.exists()),
        None
    )
    if tissue_positions_file.suffix == ".csv":
        df = pd.read_csv(tissue_positions_file, index_col=0)
    elif tissue_positions_file.suffix == ".parquet":
        df = pd.read_parquet(tissue_positions_file).set_index("barcode")

    if in_tissue:
        df = df[df["in_tissue"] > 0]
    # adjust to fullres
    df["pxl_row_in_fullres"] /= fullres_ppm
    df["pxl_col_in_fullres"] /= fullres_ppm

    # Load images
    spaceranger_spatial_path = Path(spaceranger_spatial_path)

    # image file paths
    image_files = {
    "mapped_res": mapped_image_path,
    "hires": spaceranger_spatial_path / "tissue_hires_image.png",
    "lowres": spaceranger_spatial_path / "tissue_lowres_image.png",
    }

    if use_resolution == "mapped_res" and mapped_image_path is None:
        raise ValueError("Full resolution image path must be provided.")
    if use_resolution == "mapped_res":
        print('!!! Make sure this mapped_res image is the same one you used as spaceranger input !!!')

    im = Image.open(image_files[use_resolution])
    ppm_anno = fullres_ppm * scalefactors[f"tissue_{use_resolution}_scalef"] if use_resolution != "mapped_res" else fullres_ppm # adjust resolution to the image

    # rescale image to target
    if ppm_out:
        width, height = im.size
        new_size = (int(width * ppm_out / ppm_anno), int(height * ppm_out / ppm_anno))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        ppm_anno = ppm_out

    # Convert coordinates by the same scaling
    df["pxl_col"] = df["pxl_col_in_fullres"] * ppm_anno
    df["pxl_row"] = df["pxl_row_in_fullres"] * ppm_anno

    # Convert image to array
    im = im.convert("RGBA")
    im = np.array(im)



    # Call the plotting function if plot=True
    if plot:
        plot_visium_hd(im, df, ppm_anno, int(bin_resolution), dpi=300, blowup_size_um=250,image_used=use_resolution)

    return TissueTagAnnotation(image=im,ppm=ppm_anno,positions=df)




def plot_visium_hd(
    image: np.ndarray,
    df: pd.DataFrame,
    ppm_anno: float,
    target_diameter_um: float = 16,
    dpi: int = 100,
    blowup_size_um: float = 500,
    image_used: str = None,
):
    """
    Plots Visium HD spatial data with a blowup region.

    Args:
        image: The image as a NumPy array.
        df: The DataFrame containing spatial coordinates ('pxl_col', 'pxl_row').
        ppm_anno: Pixels per micron.
        target_diameter_um: Desired marker diameter in microns.
        dpi: Figure DPI.
        blowup_size_um: Size of the blowup region in microns.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=dpi)

    # --- Main Plot (ax1) ---
    ax1.imshow(image, origin="lower")

    marker_size_pixels = target_diameter_um * ppm_anno
    marker_size_display = (
        ax1.transData.transform((marker_size_pixels, 0))[0]
        - ax1.transData.transform((0, 0))[0]
    )
    marker_size_points = marker_size_display * (72.0 / fig.dpi)
    marker_area_points_squared = marker_size_points**2

    ax1.scatter(
        df["pxl_col"],
        df["pxl_row"],
        s=marker_area_points_squared,
        color="green",
        marker=".",
        linewidths=0,
    )
    ax1.set_title(f"Visium HD Spatial Data (PPM: {ppm_anno:.2f}, Bin size:{target_diameter_um}um, Image used:{image_used})")

    # --- Blowup Region (ax2) ---

    # 1. Calculate Center of Data:
    center_x = df["pxl_col"].mean()
    center_y = df["pxl_row"].mean()

    # 2. Calculate Blowup Region Boundaries (in pixels):
    blowup_half_size_pixels = (blowup_size_um / 2) * ppm_anno
    x_min = int(center_x - blowup_half_size_pixels)
    x_max = int(center_x + blowup_half_size_pixels)
    y_min = int(center_y - blowup_half_size_pixels)
    y_max = int(center_y + blowup_half_size_pixels)

    # 3.  Handle image boundaries:
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)  # image.shape[1] is width
    y_max = min(image.shape[0], y_max)  # image.shape[0] is height

    # 4. Extract the Blowup Region from the Image:
    blowup_im = image[y_min:y_max, x_min:x_max]

    # 5. Plot the Blowup Region:
    ax2.imshow(blowup_im, origin="lower")
    ax2.set_title(f"Blowup ({blowup_size_um}um x {blowup_size_um}um)")

    # 6.  Plot points WITHIN the blowup region on the blowup plot:
    df_blowup = df[
        (df["pxl_col"] >= x_min)
        & (df["pxl_col"] < x_max)
        & (df["pxl_row"] >= y_min)
        & (df["pxl_row"] < y_max)
    ]

    df_blowup_adj = df_blowup.copy()
    df_blowup_adj["pxl_col"] -= x_min
    df_blowup_adj["pxl_row"] -= y_min

    marker_size_pixels = target_diameter_um * ppm_anno
    marker_size_display = (
        ax2.transData.transform((marker_size_pixels, 0))[0]
        - ax2.transData.transform((0, 0))[0]
    )
    marker_size_points = marker_size_display * (72.0 / fig.dpi)
    marker_area_points_squared = marker_size_points**2

    ax2.scatter(
        df_blowup_adj["pxl_col"],
        df_blowup_adj["pxl_row"],
        s=marker_area_points_squared,
        color="green",
        marker="s",
        linewidths=0,
        alpha=0.75,
    )

    # Draw a rectangle on the main plot (ax1)
    rect = matplotlib.patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
    )
    ax1.add_patch(rect)

    # Remove x and y ticks from blowup
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
from skimage.draw import disk
import cv2
from collections import OrderedDict

def gene_labels_from_adata(adata, gene_markers, tissue_tag_annotation, diameter ,override_labels=False, space_every_spots=10,normalize=True, unassigned_colour="yellow",intensity_threshold=230):
    """
    Assign labels to training spots based on gene expression from an existing AnnData object.

    Parameters
    ----------
    adata : AnnData
        Pre-loaded AnnData object containing gene expression data.
    df : pandas.DataFrame
        DataFrame containing spot coordinates.
    gene_markers : dict
        Dictionary mapping markers to genes.
    tissue_tag_annotation : TissueTagAnnotation
        Object storing label image and annotation map.
    diameter : float
        Radius of the spots.
    override_labels : boolean 
        if to remove past labels
    normalize
        if to normalise gene expression by default parametres calculated by - scanpy.pp.normalize_total()
    unassigned_colour : str, optional
        Color for unassigned labels. Default is "yellow".

    Returns
    -------
    Updated LabelAnnotation object containing the training labels.
    """
    
    if tissue_tag_annotation.label_image is not None:
        print("Label image is not empty.")
        if override_labels:
         # Initialize label image
            print("Will replace with an empty label_image.")
            tissue_tag_annotation.label_image = np.zeros((tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]), dtype=np.uint8)
        else:
            print("Will add new gene labels on top of old label_image.")
    else: # if the label_image spot is empty then create a blank one 
        tissue_tag_annotation.label_image = np.zeros((tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]), dtype=np.uint8)


    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide an annotation map.")
    else:
        tissue_tag_annotation.annotation_map = OrderedDict(tissue_tag_annotation.annotation_map)
        tissue_tag_annotation.annotation_map["unassigned"] = unassigned_colour
        tissue_tag_annotation.annotation_map.move_to_end("unassigned", last=False)


    # Filter adata to match df indices
    adata = adata[tissue_tag_annotation.positions.index.intersection(adata.obs.index)]
    r = diameter/2*tissue_tag_annotation.ppm

    # Extract coordinates
    labels = background_labels_intensity(tissue_tag_annotation.label_image.shape[:2], imarray=tissue_tag_annotation.image, r=r, intensity_threshold=intensity_threshold, space_every_spots=space_every_spots, label=1)
    mask = tissue_tag_annotation.label_image > 0
    labels[mask] = tissue_tag_annotation.label_image[mask] # add old labels if these are not empty

    if normalize:
        from scanpy.pp import normalize_total
        normalize_total(adata)

    # Assign labels based on gene expression
    for marker, gene_list in gene_markers.items():
        # Get the expected color for the marker
        marker_color = tissue_tag_annotation.annotation_map.get(marker, "N/A")
        print(f"🧬 Processing marker: '{marker}' | Color: {marker_color} | Genes: {[gene for gene, _ in gene_list]}")

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

            # Shuffle within expression levels to avoid spatial artifacts
            gene_df = gene_df.groupby("expression", group_keys=False).apply(lambda x: x.sample(frac=1))

            # Now sort by expression descending
            gene_df_sorted = gene_df.sort_values("expression", ascending=False)

            # Take top N
            actual_top_n = min(top_n, len(gene_df_sorted))
            selected_barcodes = gene_df_sorted["barcode"].iloc[:actual_top_n]

            combined_gene_indices.extend(selected_barcodes)

        # Remove duplicates and convert to a set for faster lookups later
        combined_gene_indices = set(combined_gene_indices)

        # Assign labels
        for idx, sub in enumerate(tissue_tag_annotation.annotation_map.keys()):
            if sub == marker:
                label_value = idx

        for coor in tissue_tag_annotation.positions.loc[list(combined_gene_indices), ["pxl_row", "pxl_col"]].to_numpy():
            labels[disk((coor[0], coor[1]), r)] = label_value + 1

    tissue_tag_annotation.label_image = labels



import numpy as np
from skimage.draw import disk
import cv2

def background_labels_intensity(shape, imarray, r, intensity_threshold=230, space_every_spots=10, label=1):
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
        Threshold above which pixels are considered background. Default is 200.
    every_x_spots : int, optional
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
    grid = square_grid(r, shape,space_every_spots).T

    print(imarray.shape)

    for coor in grid:
        y, x = int(coor[1]), int(coor[0])  # Ensure integer indices
        if y >= background_mask.shape[0] or x >= background_mask.shape[1]:  # Avoid out-of-bounds indexing
            continue
        if np.any(background_mask[y, x]):  # Use `.any()` if needed
            training_labels[disk((y, x), r, shape=shape)] = label
   

    return training_labels



def square_grid(spot_size, shape,space_every_spots):
    """
    Generate a square grid using vectorized operations.

    Parameters
    ----------
    spot_size : float
        Size of the spots.
    shape : tuple
        Shape of the grid (height, width).

    Returns
    -------
    numpy.ndarray
        Array containing the coordinates of the grid.
    """
    # Define step sizes
    dx = spot_size*space_every_spots  # Horizontal spacing
    dy = spot_size*space_every_spots  # Vertical spacing

    # Generate meshgrid for a square grid
    x_coords = np.arange(spot_size, shape[0] - spot_size, dx)
    y_coords = np.arange(spot_size, shape[1] - spot_size, dy)

    gx, gy = np.meshgrid(x_coords, y_coords)

    # Stack the x and y coordinates
    positions = np.vstack([gy.ravel(), gx.ravel()])

    return positions


import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from skimage.future import trainable_segmentation
from tissue_tag.annotation import rgb_from_labels,overlay_labels

def sk_rf_classifier(tissue_tag_annotation, plot=True):
    """
    A simple random forest pixel classifier from sklearn using all RGB channels as features.

    Parameters
    ----------
    tt_obj :
        tissuetag object
    plot : boolean, optional
        If to plot the loaded image. Default is True.

    Returns
    -------
    LabelAnnotation
        Predicted label map.
    """

    from skimage import filters

    print("[INFO] Initializing classifier...")

    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,  
                            sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1)  # Process all channels together

    print("[INFO] Extracting features from all RGB channels...")
    features = features_func(tissue_tag_annotation.image)  # Extract multiscale features for all channels at once

    print("[INFO] Training Random Forest classifier on RGB features...")
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
    clf = trainable_segmentation.fit_segmenter(tissue_tag_annotation.label_image, features, clf)

    print("[INFO] Predicting labels based on trained classifier...")
    predicted_labels = trainable_segmentation.predict_segmenter(features, clf)

    print("[INFO] Final label prediction completed.")

    tissue_tag_annotation.label_image = predicted_labels

    if plot:
        print("[INFO] Generating visualization...")
        labels_rgb = rgb_from_labels(tissue_tag_annotation)
        overlay_labels(tissue_tag_annotation.image, labels_rgb, alpha=0.7)
        print("[INFO] Visualization complete.")

    print("[INFO] Classification finished successfully.")
    return tissue_tag_annotation
    