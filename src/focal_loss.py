# focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default class weights from EXPERIMENTS.md
DEFAULT_WEIGHTS = {
    1: 1.0,    # tumor
    2: 1.3,    # stroma
    3: 5.7,    # lymphocyte
    4: 4.9,    # necrosis
    18: 67.6,  # blood_vessel
}

CORE_LOSS_KEY = "core_loss"

class FocalDiceLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss for semantic segmentation.
    This is designed to address class imbalance and is compatible with the trainer's
    loss interface, returning a dictionary of losses.
    """
    def __init__(self, alpha: dict = None, gamma: float = 2.0, dice_weight: float = 1.0, focal_weight: float = 1.0, **kwargs):
        """
        Args:
            alpha (dict): A dictionary mapping class_id to a weight for that class. 
                          If None, uses default weights.
            gamma (float): The focusing parameter for Focal Loss.
            dice_weight (float): Weight for the Dice loss component.
            focal_weight (float): Weight for the Focal loss component.
        """
        super(FocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        if alpha is None:
            alpha = DEFAULT_WEIGHTS
        
        # Register as buffer to move to correct device automatically
        self.register_buffer('alpha', torch.tensor([alpha.get(i, 1.0) for i in range(max(alpha.keys()) + 1)]))

    def forward(self, outputs, targets):
        """
        Calculates the combined Focal and Dice loss.

        Args:
            outputs (dict): Model output, expected to contain:
                            - 'pred_masks': Predicted masks, shape (B, N, H, W), logits.
                            - 'class_ids': Class IDs for each mask, shape (B, N).
            targets (torch.Tensor): Ground truth masks, shape (B, N, H, W).

        Returns:
            dict: A dictionary of calculated losses with a 'core_loss' key.
        """
        # Based on MultiStepMultiMasksAndIous, `outputs` is likely a list of dicts.
        # We will use the last prediction for our loss calculation.
        last_pred = outputs[-1]
        
        pred_masks = last_pred.get("pred_masks_high_res")
        class_ids = last_pred.get("class_ids") # Assumption: model forwards class_ids

        if pred_masks is None:
            raise KeyError("'pred_masks' not found in model outputs. Please check model implementation.")

        target_masks = targets.float()
        # Reshape target_masks from (B, N, H, W) to (B*N, 1, H, W)
        # where B is batch_size (expected 1 here), and N is num_objects
        target_masks = target_masks.reshape(-1, 1, target_masks.shape[-2], target_masks.shape[-1])

        # --- Focal Loss ---
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma * bce_loss
        
        # Apply class-specific weights if class_ids are available
        if class_ids is not None:
            # Move alpha to the correct device if it's not already there
            alpha_t = self.alpha[class_ids.long()].to(pred_masks.device)
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1) # Reshape for broadcasting
            focal_term = alpha_t * focal_term

        focal_loss = focal_term.mean()

        # --- Dice Loss ---
        pred_probs = torch.sigmoid(pred_masks)
        intersection = torch.sum(pred_probs * target_masks, dim=(-2, -1))
        union = torch.sum(pred_probs, dim=(-2, -1)) + torch.sum(target_masks, dim=(-2, -1))
        
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1. - dice_score.mean()

        # --- Combined Loss ---
        total_loss = (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)
        
        return {
            CORE_LOSS_KEY: total_loss,
            'focal_loss': focal_loss.detach(),
            'dice_loss': dice_loss.detach(),
        }


class OHEMDiceLoss(nn.Module):
    """
    Placeholder for Online Hard Example Mining with Dice Loss.
    This is an alternative loss function that could be implemented.
    """
    def __init__(self, top_k: float = 0.5, **kwargs):
        super(OHEMDiceLoss, self).__init__()
        self.top_k = top_k
        print("NOTE: OHEMDiceLoss is a placeholder and not fully implemented.")

    def forward(self, outputs, targets):
        """
        A placeholder forward method.
        This would typically calculate Dice loss on the hardest examples.
        """
        pred_masks = outputs.get("pred_masks")
        if pred_masks is None:
            return torch.tensor(0.0, device=targets.device)
            
        pred_probs = torch.sigmoid(pred_masks)
        target_masks = targets.float()
        
        intersection = torch.sum(pred_probs * target_masks, dim=(-2, -1))
        union = torch.sum(pred_probs, dim=(-2, -1)) + torch.sum(target_masks, dim=(-2, -1))
        
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1. - dice_score.mean()
        
        return {
            CORE_LOSS_KEY: dice_loss,
            'dice_loss': dice_loss.detach(),
        }