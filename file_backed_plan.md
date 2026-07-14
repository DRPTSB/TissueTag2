# AI Agent Task: Implement File-Backed Low-RAM Mode for TissueTag2

## Objective
Refactor the `TissueTag2` repository (specifically the `annotator_update` branch) to use a **low-RAM, file-backed matrix pipeline**. Replace in-memory NumPy/OpenCV/Pandas arrays with **Xarray datasets backed by Dask**, ensuring seamless compatibility with the existing HoloViews, Datashader, and scikit-learn infrastructure.

---

## Technical Stack Guidelines
1. **Primary Container:** Use `xarray.DataArray` or `xarray.Dataset` backed by `dask.array`. 
2. **On-Disk Format:** Use **Zarr** (`.zarr`) for storing intermediate files, labels, and pixel features. Zarr supports native chunked, parallel, file-backed reading and writing.
3. **Memory Constraint:** The code must execute without loading entire images or label matrices into RAM. Keep operations lazy until visualization or disk export.

---

## Implementation Steps

### Task 1: Audit and Dependency Setup
* **Action:** Update `pyproject.toml` or `setup.py` to include `xarray`, `dask[array]`, `zarr`, and `dask-ml` as dependencies.
* **Action:** Scan the codebase for memory-loading triggers like `.values`, `np.array()`, `cv2.imread()`, or `.compute()`. Isolate these areas for refactoring.

### Task 2: Refactor the Data Ingestion Pipeline
* **Action:** Modify the image loading function. Instead of loading the entire histology image into RAM with OpenCV or PIL, open it lazily using Xarray and Dask chunks.
* **Code pattern to implement:**
  ```python
  import xarray as xr
  # Open via zarr or a chunked reader with an explicit chunk size (e.g., 2048x2048)
  tissue_data = xr.open_dataarray("path_to_tissue.zarr", chunks={"y": 2048, "x": 2048})
  ```
* **Action:** Create an identical file-backed Zarr matrix to hold user-drawn labels/annotations (initialized to 0).

### Task 3: Adapt the HoloViews & Datashader UI
* **Action:** Trace where the tissue image and label overlays are passed to HoloViews (`hv.Image`). Ensure they receive the raw `xr.DataArray` directly, *without* converting them to NumPy.
* **Action:** Verify that `holoviews.operation.datashader.rasterize` or `datashade` handles these elements. This guarantees that only the user's visible viewport pixels are pulled into RAM dynamically during pan/zoom.

### Task 4: Refactor Interactive Annotation Writes
* **Action:** Find the backend callback function triggered when a user draws or updates an annotation (e.g., using a HoloViews stream backend).
* **Action:** Instead of rewriting a massive in-memory matrix, modify the callback to slice and update only the targeted bounding box coordinates directly onto the file-backed array.
* **Code pattern to implement:**
  ```python
  # Update a specific chunk/slice in place on disk
  labels_xarray[ymin:ymax, xmin:xmax] = user_drawn_mask
  ```

### Task 5: Refactor the Pixel Classifier (Machine Learning)
* **Action:** Locate the `scikit-learn` classifier training loop (e.g., Random Forest). 
* **Action (Training):** Modify the training step to extract and `.compute()` *only* the specific pixel coordinates where annotations exist. Do not pass the entire unannotated tissue matrix to `.fit()`.
* **Action (Prediction):** Refactor the full-image prediction pipeline to prevent RAM spikes. Use `dask_ml.wrappers.ParallelPostFit` to let the standard scikit-learn model predict the rest of the image block-by-block, saving outputs directly to disk.
* **Code pattern to implement:**
  ```python
  from dask_ml.wrappers import ParallelPostFit
  
  # Wrap the existing scikit-learn model
  wrapped_clf = ParallelPostFit(estimator=existing_sklearn_model)
  
  # Predict lazily across the entire massive image file
  predicted_chunks = wrapped_clf.predict(massive_tissue_features_xarray)
  
  # Save the results directly back to the Zarr file-backed array
  predicted_chunks.to_zarr("predicted_labels.zarr", compute=True)
  ```

---

## Definition of Done
1. A massive image (e.g., 50,000 x 50,000 pixels) can be loaded, annotated, visualized, and processed through the classifier without the Python process exceeding a predefined low RAM threshold (e.g., < 2-3 GB).
2. The user interface pan/zoom mechanics function smoothly via the Datashader/Xarray pipeline.
3. All intermediate states (features, masks) are cleanly written to disk chunks instead of accumulating in system memory.

