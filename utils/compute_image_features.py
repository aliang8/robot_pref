import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
import numpy as np

# Set torch hub cache directory
os.environ["TORCH_HOME"] = "/scr/aliang80/.cache"

def compute_dinov2_embeddings(images, batch_size=32, cache_file=None):
    """Compute DINOv2 embeddings for the images with caching support."""
    # Check if cache exists
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        embeddings = torch.load(cache_file)
        print(
            f"Loaded embeddings with shape: {embeddings.shape}, device: {embeddings.device}"
        )
        return embeddings

    print(
        f"Computing DINOv2 embeddings for {len(images)} images (this may take a while)..."
    )

    # Initialize DINOv2 model
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"DINOv2 model loaded on {device}")

    # Define image transformation for PIL images
    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    embeddings = []
    # Create tqdm progress bar
    total_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(images), batch_size),
        desc=f"Processing batches (batch size: {batch_size})",
        total=total_batches,
    ):
        batch_images = images[i : i + batch_size]

        # Convert to proper format and apply transforms
        processed_images = []
        for img in batch_images:
            # Convert torch tensor to PIL Image
            img_np = img.cpu().numpy().astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            # Apply transforms
            processed_img = transform(pil_img)
            processed_images.append(processed_img)

        # Stack into batch
        batch_tensor = torch.stack(processed_images)

        # Compute embeddings
        with torch.no_grad():
            batch_tensor = batch_tensor.to(device)
            batch_embeddings = model(batch_tensor)
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    print(
        f"Generated embeddings with shape: {all_embeddings.shape}, device: {all_embeddings.device}"
    )

    # Cache the embeddings if cache_file is provided
    if cache_file:
        print(f"Caching embeddings to {cache_file}")
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        torch.save(all_embeddings, cache_file)

    return all_embeddings
