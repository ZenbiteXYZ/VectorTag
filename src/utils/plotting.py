import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, model_name, save_dir="assets"):
    """
    Plots the training and validation loss curve.

    Args:
        history (dict): Dictionary with 'epochs', 'train_loss', 'val_loss'.
        model_name (str): Name of the model for the title and filename.
        save_dir (str): Directory to save the plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    # Use a style that is likely to exist, or fallback to default if seaborn isn't available
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')

    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    # Plot lines
    plt.plot(epochs, train_loss, linestyle='--', marker='o', markersize=4, alpha=0.6, label="Train Loss")
    plt.plot(epochs, val_loss, linewidth=2.5, marker='o', markersize=5, label="Val Loss")

    # Find and annotate best epoch
    best_idx = np.argmin(val_loss)
    best_epoch = epochs[best_idx]
    best_val = val_loss[best_idx]

    plt.scatter(best_epoch, best_val, s=150, color='gold', edgecolors='black', zorder=10, label=f"Best Val: {best_val:.4f}")

    # Annotate text
    plt.annotate(f"Min: {best_val:.4f}\n(Epoch {best_epoch})",
                 (best_epoch, best_val),
                 xytext=(0, 20), textcoords='offset points', ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.title(f"Learning Curve: {model_name}", fontsize=14, pad=15)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save
    output_path = os.path.join(save_dir, f"{model_name}_learning_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curve to: {output_path}")
