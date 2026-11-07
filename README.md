

<p align="center">
	<img src="https://github.com/DRPTSB/TissueTag2/blob/main/tissueTag2_logo.png" width="300">
</p>

# TissueTag2: a Jupyter-based tissue annotation tool for large scale and consistent analysis of spatial omics datasets

## Introduction

TissueTag2 is an interactive tissue-space annotation tool designed for seamless integration into Jupyter Notebook pipelines. TissueTag2 works on the image space and is thus compatible with any spatial modality that has a paired fluorescent and brightfield image. 

TissueTag2 consists of two major components:

1) **Jupyter-based image annotation tool**: Utilising the Bokeh Python library (http://www.bokeh.pydata.org) empowered by Datashader (https://datashader.org/index.html) and holoviews (https://holoviews.org/index.html) for pyramidal image rendering. This tool offers a streamlined annotation solution with subpixel resolution for quick interactive annotation of various image types (e.g., brightfield, fluorescence). TissueTag produces labelled images* (e.g., cortex, medulla) and logs all tissue labels, and annotation resolution and colours for reproducibility.

2) **Mapping annotations to data**: This components converts the pixel based annotations to an hexagonal grid to allow calculation of distances between morphological structures offering continuous annotation. This calculation is dependent of the grid sampling frequency and the number of grid spots that we define as a spatial neighborhood. This is foundational for calculating a morphological axis (OrganAxis, see tutorials).

We envision this tool as a starting point for other uses, so contributions and suggestions are highly appreciated!

## Installation

1) You need to install either [jupyter-lab](https://jupyter.org/install) or [jupyter-notebook](https://jupyter.org/install)
2) Clone TissueTag2 repository from GitHub:

```
git clone https://github.com/DRPTSB/TissueTag2
```

3) Install TissueTag2:

```
cd TissueTag2
pip install .
```

## How to use 
We supply four examples of usage for TissueTag2 annotations:
1) VisiumHD spatial transcriptomics - in this example we annotate a fresh-frozen mouse brain dataset [VisiumHD_annotation_tutorial](https://github.com/DRPTSB/TissueTag2/blob/main/Tutorials/demo_visium_hd_tissue_annotation_mouse_brain.ipynb).

2) Xenium in situ spatial transcriptomics - in this example we annotate a FFPE mouse brain sample dataset [Xenium_annotation_tutorial](demo_xenium_tissue_annotation.ipynb).

3) Visium spatial transcriptomics - in this example we annotate a postnatal thymus dataset by calling the major anatomical regions in multiple ways (based on marker gene expression or sparse manual annotations) then training a random forest pixel classifier for initial prediction followed by manual corrections [visium annotation tutorial](https://github.com/DRPTSB/TissueTag2/blob/main/Tutorials/demo_visium_tissue_annotation.ipynb) and migration of the annotations back to the visium anndata object [mapping annotations to visium](https://github.com/DRPTSB/TissueTag2/blob/main/Tutorials/demo_visium_annotation_to_spots.ipynb).
   We also show how to calculate a morphological axis (OrganAxis) in 2 ways. 

4) IBEX single cell multiplex protein imaging - 
   in this example we annotate a postnatal thymus image by calling the major anatomical regions and training a random forest classifier for initial prediction followed by manual corrections
   [IBEX annotation tutorial](https://github.com/DRPTSB/TissueTag2/blob/main/Tutorials/demo_flourscent_map_annotations_to_cells.ipynb). Next, we show how one can migrate these annotations to segmented cells and calulcate a morphological axis (OrganAxis) [IBEX mapping annotations tutorial](https://github.com/DRPTSB/TissueTag2/blob/main/Tutorials/demo_flourscent_map_annotations_to_cells.ipynb).
   
## Usage on a cluster vs local machine 
Bokeh interactive plotting requires communication between the notebook instance and the browser. 
We have tested the functionality of TissueTag with jupyter lab or jupyter notebooks but have not yet found a solution that works for jupyter hub.

## How to cite:
Please cite the GitHub repository for TissueTag2. 

## Previous version:
TissueTag2 builds upon the principles of its previous version [TissueTag](https://github.com/Teichlab/TissueTag) image annotation tool with major re-architectural changes to support large-scale spatial omics data and an expanded compatibility in various computing platforms while also ensuring reproducibility for a more seamless annotation experience.


