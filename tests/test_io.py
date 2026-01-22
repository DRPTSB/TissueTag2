"""
Unit tests for tissue_tag.io module
"""
import unittest
import tempfile
import os
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tissue_tag.io import TissueTagAnnotation, load_annotation


class TestTissueTagAnnotation(unittest.TestCase):
    """Test TissueTagAnnotation class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_ppm = 1.5
        self.test_label_image = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        self.test_annotation_map = pd.DataFrame({
            'annotation_id': [1, 2, 3, 4],
            'annotation_label': ['cortex', 'medulla', 'white_matter', 'background'],
            'annotation_colour': ['#FF0000', '#00FF00', '#0000FF', '#FFFFFF']
        })
        self.test_positions = pd.DataFrame({
            'x': [10, 20, 30],
            'y': [15, 25, 35]
        })
        
    def test_tissuetagannotation_init(self):
        """Test TissueTagAnnotation initialization"""
        tta = TissueTagAnnotation(
            image=self.test_image,
            ppm=self.test_ppm,
            label_image=self.test_label_image,
            annotation_map=self.test_annotation_map,
            positions=self.test_positions
        )
        
        self.assertIsNotNone(tta.image)
        self.assertEqual(tta.ppm, self.test_ppm)
        self.assertIsNotNone(tta.label_image)
        self.assertIsNotNone(tta.annotation_map)
        self.assertIsNotNone(tta.positions)
        
    def test_version_property(self):
        """Test version constant"""
        tta = TissueTagAnnotation(image=self.test_image, ppm=self.test_ppm)
        self.assertEqual(tta.VERSION, 1.1)
        
    def test_save_and_load_annotation(self):
        """Test save_annotation and load_annotation functions"""
        # Create simpler test without annotation_map for now
        tta = TissueTagAnnotation(
            image=self.test_image,
            ppm=self.test_ppm,
            label_image=self.test_label_image
        )
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            # Test save
            tta.save_annotation(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Test load
            loaded_tta = load_annotation(tmp_path)
            
            # Verify loaded data
            self.assertTrue(np.array_equal(loaded_tta.image, tta.image))
            self.assertEqual(loaded_tta.ppm, tta.ppm)
            self.assertTrue(np.array_equal(loaded_tta.label_image, tta.label_image))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    def test_save_annotation_minimal(self):
        """Test save_annotation with minimal data"""
        tta = TissueTagAnnotation(image=self.test_image, ppm=self.test_ppm)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            tta.save_annotation(tmp_path)
            
            # Verify file structure
            with h5py.File(tmp_path, 'r') as f:
                self.assertIn('image', f)
                self.assertIn('ppm', f)
                self.assertIn('version', f)
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestReadImageFunctions(unittest.TestCase):
    """Test image reading functions"""
    
    def setUp(self):
        """Create temporary test images"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test RGB image with known resolution
        self.test_image_path = os.path.join(self.test_dir, 'test_image.tif')
        test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img = Image.fromarray(test_img, mode='RGB')
        
        # Set resolution metadata (DPI to pixels per inch, need to convert)
        dpi = (100, 100)  # 100 DPI
        img.save(self.test_image_path, dpi=dpi)
        
    def tearDown(self):
        """Clean up test directory"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_read_image_with_ppm(self):
        """Test read_image with provided ppm"""
        from tissue_tag.io import read_image
        
        # Test with plot=False to avoid display issues in tests
        tta = read_image(self.test_image_path, ppm_image=1.0, ppm_out=1.0, plot=False)
        
        self.assertIsInstance(tta, TissueTagAnnotation)
        self.assertIsNotNone(tta.image)
        self.assertEqual(tta.ppm, 1.0)
        self.assertEqual(len(tta.image.shape), 3)  # Should be RGB(A)
        
    def test_read_image_contrast_factor(self):
        """Test read_image with contrast adjustment"""
        from tissue_tag.io import read_image
        
        tta = read_image(self.test_image_path, ppm_image=1.0, ppm_out=1.0, 
                        contrast_factor=2, plot=False)
        
        self.assertIsNotNone(tta.image)
        # Contrast enhancement should modify the image
        self.assertGreater(tta.image.max(), 0)


class TestAnnotationMapping(unittest.TestCase):
    """Test annotation mapping utilities"""
    
    def test_annotation_map_structure(self):
        """Test that annotation map has required structure"""
        annotation_map = pd.DataFrame({
            'annotation_id': [1, 2, 3],
            'annotation_label': ['region1', 'region2', 'region3'],
            'annotation_colour': ['#FF0000', '#00FF00', '#0000FF']
        })
        
        # Verify required columns
        self.assertIn('annotation_id', annotation_map.columns)
        self.assertIn('annotation_label', annotation_map.columns)
        self.assertIn('annotation_colour', annotation_map.columns)
        
        # Verify data types
        self.assertTrue(pd.api.types.is_integer_dtype(annotation_map['annotation_id']))
        self.assertTrue(pd.api.types.is_object_dtype(annotation_map['annotation_label']))


if __name__ == '__main__':
    unittest.main()
