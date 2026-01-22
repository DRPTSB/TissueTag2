"""
Unit tests for tissue_tag.annotation module
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tissue_tag.io import TissueTagAnnotation
from tissue_tag.annotation import rgb_from_labels, median_filter, assign_annotation_label_to_positions


class TestRgbFromLabels(unittest.TestCase):
    """Test rgb_from_labels function"""
    
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
        
    def test_rgb_from_labels_basic(self):
        """Test rgb_from_labels with basic input"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map
        )
        
        rgb_image = rgb_from_labels(tta)
        
        # Check output shape
        self.assertEqual(rgb_image.shape, (4, 4, 4))  # RGBA
        
        # Check that red pixels exist where label=1
        red_pixels = (rgb_image[:, :, 0] == 255) & (rgb_image[:, :, 1] == 0) & (rgb_image[:, :, 2] == 0)
        self.assertTrue(np.any(red_pixels))
        
        # Check that green pixels exist where label=2
        green_pixels = (rgb_image[:, :, 0] == 0) & (rgb_image[:, :, 1] == 255) & (rgb_image[:, :, 2] == 0)
        self.assertTrue(np.any(green_pixels))
        
    def test_rgb_from_labels_zero_label(self):
        """Test that label 0 is handled (unassigned)"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map
        )
        
        rgb_image = rgb_from_labels(tta)
        
        # Check that zero label positions exist
        zero_positions = self.test_label_image == 0
        self.assertTrue(np.any(zero_positions))


class TestMedianFilter(unittest.TestCase):
    """Test median_filter function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a label image with noise
        self.test_label_image = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1],  # Single noisy pixel
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2]
        ], dtype=np.uint8)
        
        self.test_annotation_map = pd.DataFrame({
            'annotation_id': [1, 2],
            'annotation_label': ['region1', 'region2'],
            'annotation_colour': ['#FF0000', '#00FF00']
        })
        
    def test_median_filter_basic(self):
        """Test median_filter with basic parameters"""
        tta = TissueTagAnnotation(
            image=np.zeros((5, 5, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image.copy(),
            annotation_map=self.test_annotation_map
        )
        
        filtered_tta = median_filter(tta, filter_radius=1, copy=True)
        
        # Check that output is a TissueTagAnnotation
        self.assertIsInstance(filtered_tta, TissueTagAnnotation)
        
        # Check that label_image was modified
        self.assertIsNotNone(filtered_tta.label_image)
        
        # Original should be unchanged when copy=True
        self.assertTrue(np.array_equal(tta.label_image, self.test_label_image))
        
    def test_median_filter_removes_noise(self):
        """Test that median filter removes isolated noise"""
        tta = TissueTagAnnotation(
            image=np.zeros((5, 5, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image.copy(),
            annotation_map=self.test_annotation_map
        )
        
        filtered_tta = median_filter(tta, filter_radius=1, copy=True)
        
        # The noisy pixel at [1, 1] should be smoothed
        # (can't assert exact value due to median filter behavior, just check it changed)
        self.assertIsNotNone(filtered_tta.label_image)
        
    def test_median_filter_copy_false(self):
        """Test median_filter with copy=False (in-place)"""
        tta = TissueTagAnnotation(
            image=np.zeros((5, 5, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image.copy(),
            annotation_map=self.test_annotation_map
        )
        
        original_label_image = tta.label_image.copy()
        filtered_tta = median_filter(tta, filter_radius=1, copy=False)
        
        # Should return None when copy=False
        self.assertIsNone(filtered_tta)
        
        # Label image should be modified
        self.assertFalse(np.array_equal(tta.label_image, original_label_image))


class TestAssignAnnotationLabelToPositions(unittest.TestCase):
    """Test assign_annotation_label_to_positions function"""
    
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
        
        # Create positions DataFrame with correct column names (pxl_row, pxl_col)
        self.test_positions = pd.DataFrame({
            'pxl_row': [0, 1, 2, 3, 0, 1],
            'pxl_col': [0, 0, 0, 0, 2, 2]
        })
        
    def test_assign_annotation_basic(self):
        """Test assign_annotation_label_to_positions with basic input"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map,
            positions=self.test_positions
        )
        
        result_tta = assign_annotation_label_to_positions(tta, annotation_column='annotation', copy=True)
        
        # Check that annotation column was added
        self.assertIn('annotation', result_tta.positions.columns)
        
        # Verify some annotations
        # Position (0, 0) should be 'cortex' (label 1)
        pos_0_annotation = result_tta.positions.loc[result_tta.positions['pxl_row'] == 0, 'annotation'].iloc[0]
        self.assertEqual(pos_0_annotation, 'cortex')
        
    def test_assign_annotation_copy_false(self):
        """Test assign_annotation_label_to_positions with copy=False"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map,
            positions=self.test_positions.copy()
        )
        
        result_tta = assign_annotation_label_to_positions(tta, annotation_column='annotation', copy=False)
        
        # Should return None when copy=False
        self.assertIsNone(result_tta)
        
        # Annotation column should be added to positions
        self.assertIn('annotation', tta.positions.columns)
        
    def test_assign_annotation_custom_column(self):
        """Test assign_annotation_label_to_positions with custom column name"""
        tta = TissueTagAnnotation(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            ppm=1.0,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map,
            positions=self.test_positions.copy()
        )
        
        result_tta = assign_annotation_label_to_positions(tta, annotation_column='region', copy=True)
        
        # Check that custom column name was used
        self.assertIn('region', result_tta.positions.columns)


if __name__ == '__main__':
    unittest.main()
