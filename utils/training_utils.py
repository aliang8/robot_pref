import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import optim
from copy import deepcopy

from utils.dataset_utils import bradley_terry_loss


def log_wandb_metrics(train_loss, val_loss, epoch, lr=None, wandb=None):
    """Log training metrics to wandb."""
    if not wandb or not wandb.run:
        return

    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
    }

    # Only add validation loss if it exists
    if val_loss is not None:
        metrics["val_loss"] = val_loss

    if lr is not None:
        metrics["learning_rate"] = lr

    # Log metrics to wandb
    wandb.log(metrics)


def evaluate_model_on_test_set(model, test_loader, device):
    """Evaluate model performance on the test set.

    Args:
        model: Trained reward model (single model or ensemble)
        test_loader: DataLoader for test data
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_acc = 0
    test_total = 0
    logpdf_values = []

    with torch.no_grad():
        for obs1, actions1, obs2, actions2, pref in tqdm(test_loader, desc="Testing"):
            # Move to device
            obs1, actions1 = obs1.to(device), actions1.to(device)
            obs2, actions2 = obs2.to(device), actions2.to(device)
            pref = pref.to(device)

            # Get reward predictions
            reward1 = model(obs1, actions1)
            reward2 = model(obs2, actions2)

            return1 = reward1.sum(dim=1)
            return2 = reward2.sum(dim=1)

            # Compute loss
            loss = bradley_terry_loss(return1, return2, pref)
            test_loss += loss.mean().item()

            # Get model predictions
            pred_pref = torch.where(
                return1 > return2, torch.ones_like(pref), torch.ones_like(pref) * 2
            )

            # Compute accuracy (prediction matches ground truth preference)
            correct = (pred_pref == pref).sum().item()
            test_acc += correct
            test_total += pref.size(0)

            # Compute logpdf
            # log of bradley terry
            bt_logits = return1 - return2
            bt_probs = torch.sigmoid(bt_logits)
            logpdf_values.append(torch.log(bt_probs).mean().item())

    avg_test_loss = (
        test_loss / len(test_loader) if len(test_loader) > 0 else float("nan")
    )
    test_accuracy = test_acc / test_total if test_total > 0 else 0
    avg_logpdf = np.mean(logpdf_values) if logpdf_values else float("nan")

    print(
        f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Avg LogPDF: {avg_logpdf:.4f}"
    )
    print(
        f"Correctly predicted {test_acc} out of {test_total} preference pairs ({test_accuracy:.2%})"
    )

    return {
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "avg_logpdf": avg_logpdf,
        "num_test_samples": test_total,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    lr=1e-4,
    wandb=None,
    is_ensemble=False,
    output_path=None,
):
    """Unified training function for both single reward models and ensembles.

    Args:
        model: Model to train (either single model or ensemble)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (can be None if no validation is needed)
        device: Device to run training on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        wandb: Optional wandb instance for logging
        is_ensemble: Whether the model is an ensemble (for logging purposes only)
        output_path: Path to save the training curve plot (if None, uses default filename)

    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    max_grad_norm = 1.0  # For gradient clipping

    # Check if we have validation data
    has_validation = val_loader is not None and len(val_loader) > 0

    print(
        f"Training {'ensemble' if is_ensemble else 'reward'} model with gradient clipping at {max_grad_norm}..."
    )
    print(f"Using learning rate: {lr}")
    if not has_validation:
        print("No validation data provided - will train without validation")

    # Clear CUDA cache before training
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Move model to device
    model = model.to(device)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        # Use a progress bar with ETA
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} (Train)",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(
            progress_bar
        ):
            # Move data to device
            obs1 = obs1.to(device, non_blocking=True)
            actions1 = actions1.to(device, non_blocking=True)
            obs2 = obs2.to(device, non_blocking=True)
            actions2 = actions2.to(device, non_blocking=True)
            pref = pref.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Compute rewards directly using the forward method
            reward1 = model(obs1, actions1)
            reward2 = model(obs2, actions2)

            if is_ensemble:
                # [B, N, T] -> [B, N]
                return1 = reward1.sum(dim=-1)
                return2 = reward2.sum(dim=-1)
                pref = pref.unsqueeze(0).repeat(model.num_models, 1)
            else:
                return1 = reward1.sum(dim=1)
                return2 = reward2.sum(dim=1)

            loss = bradley_terry_loss(return1, return2, pref)

            batch_loss = loss.mean()
            batch_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            train_loss += batch_loss.item()
            progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0

        if len(val_loader) > 0:
            val_progress = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} (Val)",
                leave=False,
                dynamic_ncols=True,
            )

            with torch.no_grad():
                for batch_idx, (obs1, actions1, obs2, actions2, pref) in enumerate(
                    val_progress
                ):
                    # Move data to device
                    obs1 = obs1.to(device, non_blocking=True)
                    actions1 = actions1.to(device, non_blocking=True)
                    obs2 = obs2.to(device, non_blocking=True)
                    actions2 = actions2.to(device, non_blocking=True)
                    pref = pref.to(device, non_blocking=True)

                    # Compute rewards directly
                    reward1 = model(obs1, actions1)
                    reward2 = model(obs2, actions2)

                    if is_ensemble:
                        # [B, N, T] -> [B, N]
                        return1 = reward1.sum(dim=-1)
                        return2 = reward2.sum(dim=-1)
                        pref = pref.unsqueeze(0).repeat(model.num_models, 1)
                    else:
                        return1 = reward1.sum(dim=1)
                        return2 = reward2.sum(dim=1)

                    batch_loss = bradley_terry_loss(return1, return2, pref)
                    batch_loss = batch_loss.mean()

                    val_loss += batch_loss.item()
                    val_progress.set_postfix({"loss": f"{batch_loss.item():.4f}"})

            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
        val_losses.append(avg_val_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

        # Log to wandb
        if wandb:
            log_wandb_metrics(avg_train_loss, avg_val_loss, epoch, lr, wandb)

    # Plot training curve
    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        if val_losses:
            plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Reward Model Training")
        plt.legend()

        # Use the provided output path or default
        plot_path = output_path if output_path else "reward_model_training.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Training curve saved to {plot_path}")

        # Log the plot to wandb
        if wandb and wandb.run:
            wandb.log({"training_curve": wandb.Image(plot_path)})

        plt.close()

    return model, train_losses, val_losses
