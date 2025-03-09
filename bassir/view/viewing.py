import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.calibration import calibration_curve


def gather_gp_inference_data(model, test_loader, device="cpu"):
    """
    Runs inference on a GP Lightning model over the test_loader
    and gathers:
      - y_true (ground truth labels)
      - y_pred_probs (predicted probabilities)
      - y_pred_vars (predictive variances for uncertainty)
    """
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    all_vars = []  # for uncertainty visualization

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            post_dist = model.likelihood(model(X_batch))
            probs = post_dist.probs  # shape (batch_size,)
            var = post_dist.variance  # shape (batch_size,)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_vars.append(var.cpu().numpy())

    # Concatenate all batches into single arrays
    y_true = np.concatenate(all_labels).astype(int)
    y_pred_probs = np.concatenate(all_probs)
    y_pred_vars = np.concatenate(all_vars)

    return y_true, y_pred_probs, y_pred_vars


def plot_roc_curve(y_true, y_pred_probs, show=True, save_path=None):
    """Plots the ROC curve and returns (fpr, tpr, roc_auc)."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    # Save BEFORE showing so the figure is still available
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fpr, tpr, roc_auc



def plot_precision_recall_curve(y_true, y_pred_probs, show=True, save_path=None):
    """Plots the Precision–Recall curve and returns (precision, recall)."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(recall, precision)
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # Save BEFORE showing so the figure is still available
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return precision, recall


def plot_confusion_matrix(y_true, y_pred_probs, threshold=0.5, show=True, save_path=None):
    """
    Plots a confusion matrix for a given threshold (default=0.5)
    and returns the confusion matrix.
    """
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Save BEFORE showing so the figure is still available
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return cm


def plot_calibration_diagram(y_true, y_pred_probs, n_bins=10, show=True, save_path=None):
    """
    Plots a reliability (calibration) diagram and returns (prob_true, prob_pred).
    The calibration curve plots the mean predicted probability against the actual fraction
    of positives in each bin.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_probs, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prob_pred, prob_true, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.legend()

    # Save BEFORE showing so the figure is still available
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return prob_true, prob_pred


def plot_variance_histogram(y_pred_vars, show=True, save_path=None):
    """
    Plots a histogram of predictive variances (uncertainty).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_pred_vars, bins=50, alpha=0.7)
    ax.set_title("Histogram of Predictive Variances")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Count")

    # Save BEFORE showing so the figure is still available
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)
