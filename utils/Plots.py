import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import tensorflow as tf 
from sklearn.metrics import confusion_matrix, classification_report




def plot_accelerometer_data(df, name):
    """
    Plot Acc-X, Acc-Y, and Acc-Z for handlebar accelerometer data over time.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Acc-X', 'Acc-Y', 'Acc-Z' columns with a time-based index.
    """
    df['NTP'] = pd.to_datetime(df['NTP'])
    df.set_index('NTP', inplace=True)
    plt.figure(figsize=(14, 7), dpi=300)
    plt.title(name)
    plt.plot(df.index, df['Acc-X'], label='Acc-X', zorder=3)
    plt.plot(df.index, df['Acc-Y'], label='Acc-Y', zorder=2)
    plt.plot(df.index, df['Acc-Z'], label='Acc-Z', zorder=1)
    plt.legend()
    plt.grid()
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    # Get the current axes and set major ticks every 120 seconds
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=120))
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.show()


def plot_anomaly_reconstruction_loss(normal_data, abnormal_data, model, threshold, y_limit=1.5):
    """
    Plot anomaly detection results showing reconstruction loss and detected anomalies.
    
    Parameters:
    -----------
    normal_data : numpy.ndarray
        Scaled normal data samples
    abnormal_data : numpy.ndarray
        Scaled abnormal data samples
    model : tensorflow.keras.Model
        Trained model, like an autoencoder or LSTM autoencoder
    threshold : float
        Threshold value for anomaly detection
    y_limit : float, optional
        Y-axis limit for the plot (default=1.5)
    
    Returns:
    --------
    tuple
        (reconstruction_loss_np, anomaly_indices) containing the reconstruction
        loss values and indices where anomalies were detected
    """
    # Combine normal and abnormal data
    all_test_data = np.vstack([normal_data, abnormal_data])
    
    # Get reconstructions and calculate loss
    all_reconstructions = model.predict(all_test_data)
    reconstruction_loss = tf.keras.losses.mae(all_reconstructions, all_test_data)
    reconstruction_loss_np = reconstruction_loss.numpy().flatten()
    
    # Find anomaly indices
    anomaly_indices = np.where(reconstruction_loss_np > threshold)[0]
    
    plt.figure(figsize=(15, 6))
    
    # Plot reconstruction loss
    plt.plot(range(len(reconstruction_loss_np)), reconstruction_loss_np, 
             'b-', linewidth=0.5, label='Reconstruction Loss')
    
    # Plot anomalies
    plt.vlines(x=anomaly_indices, 
              ymin=threshold, 
              ymax=reconstruction_loss_np[anomaly_indices],
              color='red',
              linewidth=1,
              alpha=0.5)
    
    # Add threshold line
    plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    
    # Customize the plot
    plt.xlabel('Sequence Index', fontsize=10)
    plt.ylabel('Reconstruction Loss (MAE)', fontsize=10)
    plt.ylim(0, y_limit)
    plt.grid(True, alpha=0.3)
    plt.legend(['Reconstruction Loss', 'Anomalies', 'Threshold'])
    
    plt.show()
    
    return reconstruction_loss_np, anomaly_indices

def plot_confusion_matrix(y_true, y_pred):
    """
    Creates and displays a confusion matrix visualization for binary classification results.
    
    Parameters:
        y_true (array-like): Ground truth labels (0 for normal, 1 for abnormal)
        y_pred (array-like): Predicted labels from the model
    
    Displays:
        - Confusion matrix heatmap with color-coded cells
        - Cell values showing counts of predictions
        - Classification report with precision, recall, and F1-score
    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    
    plt.imshow(confusion_mat, cmap=plt.cm.Blues)
    plt.colorbar()
    
    # Add labels with binary classes
    labels = ['0', '1']
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add numbers to cells
    threshold = confusion_mat.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_mat[i, j] > threshold else "black")
    
    plt.show()
    
    # Print classification report with binary labels
    print(classification_report(y_true, y_pred, 
                              target_names=['Normal (0)', 'Abnormal (1)'],
                              digits=3))
