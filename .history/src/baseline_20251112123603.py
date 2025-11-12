from transformers import AutoModelForImageClassification
import torch

def build_resnet50_baseline(num_classes=6):
    """
    Builds a ResNet-50 model for image classification from Hugging Face.

    Args:
        num_classes (int): The number of output classes for the classifier head.
                           Defaults to 6 for the BCSS dataset classes.

    Returns:
        A PyTorch model instance.
    """
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Allow replacing the classifier head
    )
    return model

if __name__ == '__main__':
    # Quick smoke test to verify the model can be built and run on a dummy input.
    print("Building ResNet-50 baseline model...")
    baseline_model = build_resnet50_baseline()
    print("Model built successfully.")

    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Perform a forward pass
    with torch.no_grad():
        outputs = baseline_model(dummy_input)
        logits = outputs.logits

    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (1, 6)
    print("Smoke test passed: Model output shape is correct.")