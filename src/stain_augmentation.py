# src/stain_augmentation.py
"""
Stain Normalization and Augmentation Module.

This module implements the Macenko stain normalization method and provides
a transform for use in data loading pipelines. The core logic is adapted
from the StainTools library by Peter Byfield:
https://github.com/Peter554/StainTools

Dependencies were consolidated into this single file to avoid installation
issues in the target environment. This requires `scikit-learn`, which is
assumed to be present.
"""

import numpy as np
import cv2 as cv
from sklearn.linear_model import Lasso
from PIL import Image

# --- Utility functions from StainTools ---

def is_uint8_image(I):
    if not isinstance(I, np.ndarray) or I.dtype != np.uint8:
        return False
    return True

def convert_RGB_to_OD(I):
    mask = (I == 0)
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255.0), 1e-6)

def convert_OD_to_RGB(OD):
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)

def normalize_matrix_rows(A):
    return A / np.linalg.norm(A, axis=1)[:, None]

def get_concentrations(I, stain_matrix, regularizer=0.01):
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    # Using scikit-learn's Lasso as a replacement for spams.lasso
    # solver='cd' is coordinate descent, similar to spams
    lasso = Lasso(alpha=regularizer, positive=True, fit_intercept=False)
    lasso.fit(stain_matrix.T, OD.T)
    return lasso.coef_.T

# --- Tissue Masking from StainTools ---

class ABCTissueLocator:
    @staticmethod
    def get_tissue_mask(I):
        raise NotImplementedError

class LuminosityThresholdTissueLocator(ABCTissueLocator):
    @staticmethod
    def get_tissue_mask(I, luminosity_threshold=0.8):
        assert is_uint8_image(I), "Image should be RGB uint8."
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0
        mask = L < luminosity_threshold
        if mask.sum() == 0:
            # Fallback: return a mask of all true if no tissue is found
            return np.full(I.shape[:2], True, dtype=bool)
        return mask

# --- Stain Extraction from StainTools ---

class ABCStainExtractor:
    @staticmethod
    def get_stain_matrix(I):
        raise NotImplementedError

class MacenkoStainExtractor(ABCStainExtractor):
    @staticmethod
    def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):
        assert is_uint8_image(I), "Image should be RGB uint8."
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]

        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
        V = V[:, [2, 1]]

        if V[0, 0] < 0: V[:, 0] *= -1
        if V[0, 1] < 0: V[:, 1] *= -1

        That = np.dot(OD, V)
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)

        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])

        return normalize_matrix_rows(HE)

# --- Stain Normalizer and Augmentor ---

class StainNormalizer:
    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None

    def fit(self, target):
        self.stain_matrix_target = MacenkoStainExtractor.get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))

    def transform(self, I):
        if self.stain_matrix_target is None:
            # Fit to a default target if not already fitted.
            # This is a fallback, it's better to fit to a reference image.
            # For now, we will fit to the image itself if no target is set.
            self.fit(I)

        stain_matrix_source = MacenkoStainExtractor.get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (self.maxC_target / maxC_source)
        
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)

class StainAugmentor:
    def __init__(self, alpha_range=0.2, beta_range=0.2):
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def augment(self, concentrations):
        alpha = 1 + np.random.uniform(-self.alpha_range, self.alpha_range, size=concentrations.shape[1])
        beta = np.random.uniform(-self.beta_range, self.beta_range, size=concentrations.shape[1])
        return concentrations * alpha + beta

from PIL import Image

class StainAugmentationTransform:
    """
    A transform compatible with the 'training.dataset.transforms' pipeline.
    It takes a PIL image, applies Macenko stain normalization/augmentation,
    and returns a PIL image.
    """
    def __init__(self, normalize: bool = True, augment: bool = True):
        self.normalize = normalize
        self.augment = augment
        self.normalizer = StainNormalizer()
        self.augmentor = StainAugmentor()
        self.target_fitted = False

    def __call__(self, datapoint: dict, **kwargs) -> dict:
        """
        Args:
            datapoint (dict): A dictionary containing the image and other data.
                              The image is expected under the key 'image'.

        Returns:
            dict: The modified datapoint.
        """
        if hasattr(datapoint, 'image'):
            img_pil = datapoint.image
            img_np = np.array(img_pil)
            
            if not self.target_fitted:
                self.normalizer.fit(img_np)
                self.target_fitted = True

            if self.normalize:
                stain_matrix_source = MacenkoStainExtractor.get_stain_matrix(img_np)
                concentrations = get_concentrations(img_np, stain_matrix_source)
                
                if self.augment:
                    concentrations = self.augmentor.augment(concentrations)
                
                transformed_img = 255 * np.exp(-1 * np.dot(concentrations, self.normalizer.stain_matrix_target))
                transformed_img_np = transformed_img.reshape(img_np.shape).astype(np.uint8)
            
            else:
                transformed_img_np = img_np
            
            datapoint.image = Image.fromarray(transformed_img_np)

        return datapoint

