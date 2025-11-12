from transformers import AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader
from src.dataset import BCSSDataset

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
    # Example of a simple training loop
    print("\n--- Starting simple training loop example ---")
    try:
        import torch.optim as optim

        # 1. Load dataset
        print("Loading training data...")
        # Note: BCSSDataset may require data to be downloaded first.
        # See project README.
        train_dataset = BCSSDataset(split='train')
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        print(f"Training data loaded: {len(train_dataset)} samples.")

        # 2. Setup model, optimizer, and loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        baseline_model.to(device)
        optimizer = optim.Adam(baseline_model.parameters(), lr=1e-4)
        # Use CrossEntropyLoss for multi-class classification.
        # The label should be the class index.
        criterion = torch.nn.CrossEntropyLoss()

        # 3. Simple training loop (for demonstration purposes)
        num_epochs = 1
        print(f"Training on {device} for {num_epochs} epoch(s)...")
        baseline_model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # For this example, we'll use a simple strategy:
                # Classify the image based on the presence of a tumor (class 1).
                # Label is 1 if tumor is present, 0 otherwise.
                inputs = data['image'].to(device)
                # Create a binary label: 1 if tumor (class_id 1) is present, 0 otherwise.
                labels = torch.tensor([1 if 1 in uc else 0 for uc in data['unique_classes']], dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = baseline_model(inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9: # Print every 10 mini-batches
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
                # This is a demo, so we'll just run a few batches.
                if i > 20:
                    print("Stopping training loop example after a few batches.")
                    break

        print("Finished simple training loop example.")

    except ImportError as e:
        print(f"\nCould not run training example due to missing import: {e}")
        print("Please ensure all dependencies are installed and the script is run from the project root.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nCould not run training example: {e}")
        print("Please ensure the BCSS dataset is downloaded and configured correctly as per the README.")
