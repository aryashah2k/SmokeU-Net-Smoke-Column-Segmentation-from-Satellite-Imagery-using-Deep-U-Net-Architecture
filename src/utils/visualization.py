import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_batch(imgs, masks, preds=None, color_pred=[0, 0, 1], color_true=[0, 1, 0], batch_size=32):
    """
    Plots a batch of images with their masks and predictions.
    
    Args:
        imgs: Batch of images
        masks: Batch of ground truth masks
        preds: Batch of predictions (optional)
        color_pred: RGB color for prediction contours
        color_true: RGB color for ground truth contours
        batch_size: Number of images to display
    """
    plt.figure(figsize=(20, 10))
    
    for i in range(batch_size):
        plt.subplot(4, 8, i + 1)
        img = imgs[i, ...].permute(1, 2, 0).numpy()
        mask = masks[i, ...].permute(1, 2, 0).numpy()
        
        img = np.clip(img, 0, 1)
        
        true_contour_mask = cv2.Canny(mask[:, :, 0].astype(np.uint8), 0, 1)
        img_with_contour = img.copy()
        img_with_contour[true_contour_mask > 0] = color_true
        
        if preds is not None:
            pred = preds[i, ...].permute(1, 2, 0).numpy()
            pred_contour_mask = cv2.Canny(pred[:, :, 0].astype(np.uint8), 0, 1)
            img_with_contour[pred_contour_mask > 0] = color_pred
            
        plt.imshow(img_with_contour)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_metrics(history_df, train_label, val_label, y_label, title):
    """
    Plots training and validation metrics over epochs.
    
    Args:
        history_df: DataFrame containing training history
        train_label: Column name for training metric
        val_label: Column name for validation metric
        y_label: Label for y-axis
        title: Plot title
    """
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(history_df["epoch"], history_df[train_label], 
             linestyle='-', color='gray', 
             label=f"{y_label} in the Training Set")
    plt.plot(history_df["epoch"], history_df[val_label], 
             linestyle='-', color='#59B0F0', 
             label=f"{y_label} in the Validation Set")
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.xticks(history_df["epoch"])
    plt.title(title)
    plt.grid(True, linestyle='-', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()