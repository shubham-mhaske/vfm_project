# src/iterative_refinement.py
"""
Iterative mask refinement for SAM2.
Feeds the predicted mask back to SAM2 for refinement, improving boundary accuracy.
"""
import numpy as np
import torch
import torch.nn.functional as F

# SAM2 expects mask_input at this resolution
SAM2_MASK_INPUT_SIZE = 256


def predict_with_refinement(
    predictor,
    image: np.ndarray,
    prompts: dict,
    prompt_type: str = 'box',
    use_neg_points: bool = True,
    num_iterations: int = 2,
    refinement_threshold: float = 0.5
) -> np.ndarray:
    """
    Apply iterative mask refinement to improve segmentation quality.
    
    The function:
    1. Gets initial prediction with prompts
    2. Feeds the predicted mask back to SAM2 as mask_input
    3. Repeats for num_iterations
    
    Args:
        predictor: The SAM2ImagePredictor instance.
        image (np.ndarray): The input image as a numpy array.
        prompts (dict): A dictionary of prompts generated from the mask.
        prompt_type (str): The type of prompt to use ('box', 'centroid', 'multi_point').
        use_neg_points (bool): Whether to include negative points.
        num_iterations (int): Number of refinement iterations (default 2).
        refinement_threshold (float): Threshold for binarizing mask (default 0.5).
    
    Returns:
        np.ndarray: The refined prediction mask.
    """
    from .sam_segmentation import get_predicted_mask_from_prompts
    
    # Set image once
    predictor.set_image(image)
    
    # Get initial prediction
    mask_logits, _, _ = get_predicted_mask_from_prompts(
        predictor, image, prompts, prompt_type, use_neg_points
    )
    
    current_mask = mask_logits
    
    # Iterative refinement
    for iteration in range(num_iterations - 1):
        # Convert mask to proper format for SAM2's mask_input
        # SAM2 expects mask_input of shape [1, 256, 256] with logits (numpy array)
        h, w = current_mask.shape
        
        # Convert to logits-like format (SAM2 expects logits, not binary)
        # Scale binary mask to approximate logits range
        mask_input = current_mask.astype(np.float32)
        mask_input = (mask_input - 0.5) * 20  # Scale to logit range [-10, 10]
        mask_input = mask_input[np.newaxis, np.newaxis, :, :]  # [1, 1, H, W]
        mask_input = torch.from_numpy(mask_input).float()
        
        # Resize to SAM2's expected 256x256 resolution
        mask_input = F.interpolate(
            mask_input, 
            size=(SAM2_MASK_INPUT_SIZE, SAM2_MASK_INPUT_SIZE), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert back to numpy array with shape [1, 256, 256] as SAM2 expects
        mask_input = mask_input.squeeze(0).numpy()  # [1, 256, 256]
        
        # Get refined prediction with mask input
        refined_mask = _predict_with_mask_input(
            predictor, prompts, prompt_type, use_neg_points, mask_input
        )
        
        current_mask = refined_mask
    
    # Return binary mask
    return (current_mask > refinement_threshold).astype(np.uint8)


def _predict_with_mask_input(
    predictor,
    prompts: dict,
    prompt_type: str,
    use_neg_points: bool,
    mask_input: np.ndarray
) -> np.ndarray:
    """
    Internal function to get prediction with mask input.
    
    Args:
        mask_input: numpy array of shape [1, 256, 256] with logits
    """
    point_coords, point_labels, box = None, None, None
    
    if prompt_type == 'centroid':
        point_coords, point_labels = prompts['centroid']
    elif prompt_type == 'multi_point':
        point_coords, point_labels = prompts['multi_point']
    elif prompt_type == 'box':
        box = prompts['box']
    
    if use_neg_points and 'neg_points' in prompts:
        neg_coords, neg_labels = prompts['neg_points']
        if point_coords is not None:
            point_coords = np.concatenate([point_coords, neg_coords], axis=0)
            point_labels = np.concatenate([point_labels, neg_labels], axis=0)
        else:
            point_coords, point_labels = neg_coords, neg_labels
    
    device = predictor.device
    if device == "mps":
        autocast_context = torch.autocast("mps", dtype=torch.float16)
    elif device == "cuda":
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_context = torch.no_grad()
    
    with autocast_context:
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,  # numpy array [1, 256, 256]
            multimask_output=False
        )
    
    return masks[0]


def predict_with_tta_and_refinement(
    predictor,
    image: np.ndarray,
    prompts: dict,
    prompt_type: str = 'box',
    use_neg_points: bool = True,
    num_augmentations: int = 4,
    num_refinement_iterations: int = 2
) -> np.ndarray:
    """
    Combine TTA with iterative refinement for best results.
    
    Strategy:
    1. Get TTA ensemble prediction
    2. Refine the ensemble with mask feedback
    
    Args:
        predictor: The SAM2ImagePredictor instance.
        image (np.ndarray): The input image as a numpy array.
        prompts (dict): A dictionary of prompts generated from the mask.
        prompt_type (str): The type of prompt to use.
        use_neg_points (bool): Whether to include negative points.
        num_augmentations (int): Number of TTA augmentations (1-4).
        num_refinement_iterations (int): Number of refinement iterations.
    
    Returns:
        np.ndarray: The refined TTA prediction mask.
    """
    from .tta_utils import predict_with_tta
    
    # Step 1: Get TTA ensemble prediction
    tta_mask = predict_with_tta(
        predictor, image, prompts, prompt_type, use_neg_points, num_augmentations
    )
    
    if num_refinement_iterations <= 1:
        return tta_mask
    
    # Step 2: Refine the TTA prediction
    predictor.set_image(image)
    
    current_mask = tta_mask.astype(np.float32)
    
    for iteration in range(num_refinement_iterations - 1):
        h, w = current_mask.shape
        
        # Convert to logits format
        mask_input = (current_mask - 0.5) * 20
        mask_input = mask_input[np.newaxis, np.newaxis, :, :]
        mask_input = torch.from_numpy(mask_input).float()
        
        # Resize to SAM2's expected 256x256 resolution
        mask_input = F.interpolate(
            mask_input,
            size=(SAM2_MASK_INPUT_SIZE, SAM2_MASK_INPUT_SIZE),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert back to numpy array with shape [1, 256, 256] as SAM2 expects
        mask_input = mask_input.squeeze(0).numpy()  # [1, 256, 256]
        
        refined_mask = _predict_with_mask_input(
            predictor, prompts, prompt_type, use_neg_points, mask_input
        )
        
        current_mask = refined_mask
    
    return (current_mask > 0.5).astype(np.uint8)
