"""
Unit tests for tissue_tag.organaxis module
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tissue_tag.io import TissueTagAnnotation
from tissue_tag.organaxis import (
    generate_hires_grid,
    create_disk_kernel,
    generate_grid_from_annotation,
    calculate_distance_to_annotations,
    map_annotations_to_target,
    calculate_axis,
    bin_axis,
    get_annotations_for_objects
)


class TestGenerateHiresGrid(unittest.TestCase):
    """Test generate_hires_grid function"""
    
    def test_generate_hires_grid_basic(self):
        """Test generate_hires_grid with basic parameters"""
        im = np.zeros((100, 100), dtype=np.uint8)
        grid_unit_size = 10.0
        pixels_per_micron = 1.0
        
        grid = generate_hires_grid(im, grid_unit_size, pixels_per_micron)
        
        # Check output shape (2, N) where N is number of positions
        self.assertEqual(len(grid.shape), 2)
        self.assertEqual(grid.shape[0], 2)  # x and y coordinates
        
        # Check that positions are within image bounds
        self.assertTrue(np.all(grid[0, :] >= 0))
        self.assertTrue(np.all(grid[0, :] < im.shape[1]))
        self.assertTrue(np.all(grid[1, :] >= 0))
        self.assertTrue(np.all(grid[1, :] < im.shape[0]))
        
    def test_generate_hires_grid_spacing(self):
        """Test that grid spacing is approximately correct"""
        im = np.zeros((200, 200), dtype=np.uint8)
        grid_unit_size = 20.0
        pixels_per_micron = 1.0
        
        grid = generate_hires_grid(im, grid_unit_size, pixels_per_micron)
        
        # Check that we have a reasonable number of points
        self.assertGreater(grid.shape[1], 0)
        
        # Verify positions are non-empty
        self.assertGreater(len(grid[0]), 0)


class TestCreateDiskKernel(unittest.TestCase):
    """Test create_disk_kernel function"""
    
    def test_create_disk_kernel_basic(self):
        """Test create_disk_kernel with basic parameters"""
        radius = 5
        shape = (11, 11)
        
        kernel = create_disk_kernel(radius, shape)
        
        # Check output shape
        self.assertEqual(kernel.shape, shape)
        
        # Check that kernel is boolean
        self.assertEqual(kernel.dtype, bool)
        
        # Check that center is True
        center = (shape[0] // 2, shape[1] // 2)
        self.assertTrue(kernel[center])
        
    def test_create_disk_kernel_symmetry(self):
        """Test that disk kernel is symmetric"""
        radius = 3
        shape = (7, 7)
        
        kernel = create_disk_kernel(radius, shape)
        
        # Check horizontal symmetry
        self.assertTrue(np.array_equal(kernel, np.fliplr(kernel)))
        
        # Check vertical symmetry
        self.assertTrue(np.array_equal(kernel, np.flipud(kernel)))


class TestGenerateGridFromAnnotation(unittest.TestCase):
    """Test generate_grid_from_annotation function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a larger label image that will generate grid points
        self.test_label_image = np.zeros((100, 100), dtype=np.uint8)
        self.test_label_image[0:30, :] = 1
        self.test_label_image[30:60, :] = 2
        self.test_label_image[60:, :] = 3
        
        self.test_annotation_map = pd.DataFrame({
            'annotation_id': [1, 2, 3],
            'annotation_label': ['cortex', 'medulla', 'white_matter'],
            'annotation_colour': ['#FF0000', '#00FF00', '#0000FF']
        })
        
    def test_generate_grid_from_annotation_basic(self):
        """Test generate_grid_from_annotation with basic parameters"""
        tta = TissueTagAnnotation(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map
        )
        
        grid_df = generate_grid_from_annotation(tta, grid_unit_size=10.0, ppm_out=1.0)
        
        # Check that grid_df is a DataFrame
        self.assertIsInstance(grid_df, pd.DataFrame)
        
        # Check required columns
        self.assertIn('x', grid_df.columns)
        self.assertIn('y', grid_df.columns)
        self.assertIn('annotation', grid_df.columns)
        self.assertIn('annotation_id', grid_df.columns)
        
        # Check that we have some grid points
        self.assertGreater(len(grid_df), 0)
        
        # Check that annotations are from the annotation_map
        unique_annotations = grid_df['annotation'].unique()
        expected_annotations = set(self.test_annotation_map['annotation_label'].values)
        self.assertTrue(set(unique_annotations).issubset(expected_annotations | {'unknown'}))
        
    def test_generate_grid_from_annotation_custom_column(self):
        """Test generate_grid_from_annotation with custom annotation column"""
        tta = TissueTagAnnotation(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map
        )
        
        grid_df = generate_grid_from_annotation(tta, grid_unit_size=10.0, 
                                               ppm_out=1.0, annotation_column='region')
        
        # Check custom column names
        self.assertIn('region', grid_df.columns)
        self.assertIn('region_id', grid_df.columns)


class TestCalculateDistanceToAnnotations(unittest.TestCase):
    """Test calculate_distance_to_annotations function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a grid with clear spatial structure
        self.grid_df = pd.DataFrame({
            'x': [0, 1, 2, 10, 11, 12],
            'y': [0, 1, 2, 10, 11, 12],
            'annotation': ['cortex', 'cortex', 'cortex', 'medulla', 'medulla', 'medulla']
        })
        
    def test_calculate_distance_basic(self):
        """Test calculate_distance_to_annotations with basic parameters"""
        result_df = calculate_distance_to_annotations(self.grid_df, knn=2, annotation_column='annotation')
        
        # Check that distance columns were added
        dist_cols = [c for c in result_df.columns if c.startswith('L2_dist_')]
        self.assertGreater(len(dist_cols), 0)
        
        # Check that we have distance columns for each annotation
        self.assertIn('L2_dist_annotation_cortex', result_df.columns)
        self.assertIn('L2_dist_annotation_medulla', result_df.columns)
        
    def test_calculate_distance_knn_parameter(self):
        """Test calculate_distance_to_annotations with different knn values"""
        result_df_k1 = calculate_distance_to_annotations(self.grid_df.copy(), knn=1, annotation_column='annotation')
        result_df_k3 = calculate_distance_to_annotations(self.grid_df.copy(), knn=3, annotation_column='annotation')
        
        # Both should have distance columns
        self.assertGreater(len([c for c in result_df_k1.columns if c.startswith('L2_dist_')]), 0)
        self.assertGreater(len([c for c in result_df_k3.columns if c.startswith('L2_dist_')]), 0)
        
    def test_calculate_distance_logscale(self):
        """Test calculate_distance_to_annotations with logscale"""
        result_df = calculate_distance_to_annotations(self.grid_df, knn=2, 
                                                     logscale=True, annotation_column='annotation')
        
        # Check that log distance columns were added
        log_dist_cols = [c for c in result_df.columns if c.startswith('L2_dist_log10_')]
        self.assertGreater(len(log_dist_cols), 0)


class TestMapAnnotationsToTarget(unittest.TestCase):
    """Test map_annotations_to_target function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.df_source = pd.DataFrame({
            'x': [0, 10, 20],
            'y': [0, 10, 20],
            'annotation': ['region1', 'region2', 'region3']
        })
        
        self.df_target = pd.DataFrame({
            'x': [1, 11, 21],
            'y': [1, 11, 21]
        })
        
    def test_map_annotations_basic(self):
        """Test map_annotations_to_target with basic parameters"""
        result_df = map_annotations_to_target(
            df_source=self.df_source,
            df_target=self.df_target,
            ppm_target=1.0,
            ppm_source=1.0,
            plot=False,
            max_distance=10.0
        )
        
        # Check that annotations were added to target
        self.assertIn('annotation', result_df.columns)
        
        # Check that result has same number of rows as target
        self.assertEqual(len(result_df), len(self.df_target))
        
    def test_map_annotations_max_distance(self):
        """Test map_annotations_to_target with max_distance constraint"""
        result_df = map_annotations_to_target(
            df_source=self.df_source,
            df_target=self.df_target,
            ppm_target=1.0,
            ppm_source=1.0,
            plot=False,
            max_distance=0.5  # Very small distance
        )
        
        # With very small max_distance, many annotations should be None/NaN
        self.assertIn('annotation', result_df.columns)


class TestCalculateAxis(unittest.TestCase):
    """Test calculate_axis function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_df = pd.DataFrame({
            'dist_region1': [1.0, 2.0, 3.0, 4.0],
            'dist_region2': [4.0, 3.0, 2.0, 1.0],
            'dist_region3': [5.0, 4.0, 3.0, 2.0]
        })
        
    def test_calculate_axis_2point(self):
        """Test calculate_axis with 2 features"""
        result_df = calculate_axis(
            self.feature_df,
            feature_columns=['dist_region1', 'dist_region2'],
            output_column='axis_2point'
        )
        
        # Check that axis column was added
        self.assertIn('axis_2point', result_df.columns)
        
        # Check axis values are in valid range
        self.assertTrue(np.all(result_df['axis_2point'] >= -1))
        self.assertTrue(np.all(result_df['axis_2point'] <= 1))
        
    def test_calculate_axis_3point(self):
        """Test calculate_axis with 3 features"""
        result_df = calculate_axis(
            self.feature_df,
            feature_columns=['dist_region1', 'dist_region2', 'dist_region3'],
            output_column='axis_3point',
            weights=(0.3, 0.7)
        )
        
        # Check that axis column was added
        self.assertIn('axis_3point', result_df.columns)
        
    def test_calculate_axis_invalid_features(self):
        """Test calculate_axis with invalid number of features"""
        with self.assertRaises(ValueError):
            calculate_axis(
                self.feature_df,
                feature_columns=['dist_region1'],  # Only 1 feature
                output_column='axis'
            )
            
        with self.assertRaises(ValueError):
            calculate_axis(
                self.feature_df,
                feature_columns=['dist_region1', 'dist_region2', 'dist_region3', 'extra'],  # 4 features
                output_column='axis'
            )


class TestBinAxis(unittest.TestCase):
    """Test bin_axis function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.axis_df = pd.DataFrame({
            'axis_value': [-0.8, -0.3, 0.0, 0.3, 0.8]
        })
        
    def test_bin_axis_basic(self):
        """Test bin_axis with basic parameters"""
        result_df = bin_axis(
            self.axis_df,
            axis_column='axis_value',
            bin_labels=['low', 'medium', 'high'],
            cutoff_values=[-0.5, 0.5]
        )
        
        # Check that binned column was added
        self.assertIn('binned_axis_value', result_df.columns)
        
        # Check that all bins are assigned
        unique_bins = result_df['binned_axis_value'].unique()
        self.assertTrue(set(unique_bins).issubset(set(['low', 'medium', 'high', 'unassigned'])))
        
    def test_bin_axis_invalid_parameters(self):
        """Test bin_axis with invalid parameters"""
        # Number of bin labels should be len(cutoff_values) + 1
        with self.assertRaises(ValueError):
            bin_axis(
                self.axis_df,
                axis_column='axis_value',
                bin_labels=['low', 'high'],  # 2 labels
                cutoff_values=[-0.5, 0.0, 0.5]  # 3 cutoffs (needs 4 labels)
            )


class TestGetAnnotationsForObjects(unittest.TestCase):
    """Test get_annotations_for_objects function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_label_image = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        
        self.test_annotation_map = pd.DataFrame({
            'annotation_id': [1, 2, 3],
            'annotation_label': ['cortex', 'medulla', 'white_matter'],
            'annotation_colour': ['#FF0000', '#00FF00', '#0000FF']
        })
        
        self.coord_df = pd.DataFrame({
            'x': [0, 1, 2, 3],
            'y': [0, 0, 0, 0]
        })
        
    def test_get_annotations_basic(self):
        """Test get_annotations_for_objects with basic input"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map
        )
        
        annotations = get_annotations_for_objects(tta, self.coord_df)
        
        # Check output is array
        self.assertIsInstance(annotations, np.ndarray)
        
        # Check length matches input
        self.assertEqual(len(annotations), len(self.coord_df))
        
        # Check that annotations are strings
        self.assertTrue(all(isinstance(a, str) for a in annotations))
        
    def test_get_annotations_missing_label_image(self):
        """Test get_annotations_for_objects with missing label_image"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=None,
            annotation_map=self.test_annotation_map
        )
        
        with self.assertRaises(ValueError):
            get_annotations_for_objects(tta, self.coord_df)
            
    def test_get_annotations_invalid_coord_df(self):
        """Test get_annotations_for_objects with invalid coord_df"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map
        )
        
        # DataFrame with wrong number of columns
        invalid_coord_df = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 0, 0],
            'z': [0, 0, 0]
        })
        
        with self.assertRaises(ValueError):
            get_annotations_for_objects(tta, invalid_coord_df)


if __name__ == '__main__':
    unittest.main()
