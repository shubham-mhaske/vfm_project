# src/tta_utils.py
import numpy as np
import torch

def predict_with_tta(
    predictor,
    image: np.ndarray,
    prompts: dict,
    prompt_type: str = 'box',
    use_neg_points: bool = True,
    num_augmentations: int = 4
) -> np.ndarray:
    """
    Apply test-time augmentation to get a more robust prediction.
    The function applies a set of augmentations to the image and prompts,
    gets a prediction for each, de-augments the prediction, and averages
    the results.

    Args:
        predictor: The SAM2ImagePredictor instance.
        image (np.ndarray): The input image as a numpy array.
        prompts (dict): A dictionary of prompts generated from the mask.
        prompt_type (str): The type of prompt to use ('box', 'centroid', 'multi_point').
        use_neg_points (bool): Whether to include negative points.
        num_augmentations (int): Number of augmentations to use (1-4).

    Returns:
        np.ndarray: The ensembled (averaged) prediction mask.
    """
    h, w, _ = image.shape
    augmented_masks = []

    # A helper to transform coordinates
    def transform_coords(coords, transform_type):
        if coords is None:
            return None
        
        new_coords = coords.copy()
        if transform_type == 'hflip':
            new_coords[:, 0] = w - new_coords[:, 0]
        elif transform_type == 'vflip':
            new_coords[:, 1] = h - new_coords[:, 1]
        elif transform_type == 'rot90':
            new_coords = new_coords[:, [1, 0]] # Swap x and y
            new_coords[:, 1] = w - new_coords[:, 1] # New y is w - old x
        return new_coords

    augmentations = ['original', 'hflip', 'vflip', 'rot90'][:num_augmentations]

    for aug_type in augmentations:
        aug_image = image.copy()
        aug_prompts = {k: (v[0].copy(), v[1].copy()) if isinstance(v, tuple) else v.copy() for k, v in prompts.items()}

        # Augment image (use .copy() to ensure contiguous array with positive strides)
        if aug_type == 'hflip':
            aug_image = np.fliplr(aug_image).copy()
        elif aug_type == 'vflip':
            aug_image = np.flipud(aug_image).copy()
        elif aug_type == 'rot90':
            aug_image = np.rot90(aug_image).copy()

        # Augment prompts
        for p_type, p_val in aug_prompts.items():
            if 'point' in p_type:
                aug_prompts[p_type] = (transform_coords(p_val[0], aug_type), p_val[1])
            elif p_type == 'box':
                # Box is [[x0,y0],[x1,y1]]. Need to handle min/max after transform.
                tl, br = p_val
                coords = np.array([tl, br])
                new_coords = transform_coords(coords, aug_type)
                aug_prompts['box'] = np.array([np.min(new_coords, axis=0), np.max(new_coords, axis=0)])

        # Predict
        from .sam_segmentation import get_predicted_mask_from_prompts
        mask, _, _ = get_predicted_mask_from_prompts(
            predictor, aug_image, aug_prompts, prompt_type, use_neg_points
        )

        # De-augment mask
        de_aug_mask = mask
        if aug_type == 'hflip':
            de_aug_mask = np.fliplr(de_aug_mask)
        elif aug_type == 'vflip':
            de_aug_mask = np.flipud(de_aug_mask)
        elif aug_type == 'rot90':
            de_aug_mask = np.rot90(de_aug_mask, k=-1)
        
        augmented_masks.append(de_aug_mask.astype(np.float32))

    # Ensemble by averaging
    ensembled_mask = np.mean(augmented_masks, axis=0)

    # Return a binary mask using a 0.5 threshold on the averaged probabilities
    return (ensembled_mask > 0.5).astype(np.uint8)
