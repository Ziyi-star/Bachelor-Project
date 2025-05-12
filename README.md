# Bachelor Project: Anomaly Detection Techniques for Cyclist Curb Recognition using Traditional Machine Learning und Deep Learning Techniques for Time Series Classification

## Overview

This repository contains the code and documentation for my Bachelor's project, focusing on the application of deep learning techniques to time series classification. The objective is to explore and evaluate one class SVM, Autoencoder and LSTM Autoencoder architectures to classify in time-series data.

## General Information
This repository was created as part of the bachelor project *Anomaly Detection Techniques for Cyclist Curb Recognition*.

## Project Structure

- `data/` – Contains datasets used for training and evaluation, including preprocessed time series data.  
- `models/` – Different Models Autoencoder, LSTM Autoencoder, one class SVM
- `notebooks/` – Jupyter notebooks for exploratory data analysis, model training, and evaluation.
- `results/` – Evaluation Results, Plots and other visualizations  
- `utils/` – Utility functions for this project  
- `LICENSE` – Information about the terms of use for this repository  


## Key Features

- **Data Preprocessing**: Handling missing values, normalization, label, and windowing of time series data.
- **Model Implementations**:
  - One Class SVM
  - Autoencoder
  - Long Short-Term Memory (LSTM)

- **Training Pipeline**: Modular training scripts with support for early stopping, learning rate scheduling, and feintuning.
- **Evaluation Metrics**: MAE
- **Visualization**: Plotting training curves, prediction vs. actual values, and attention weights (for Transformer models).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
git clone https://github.com/Ziyi-star/Bachelor-Project.git
cd Bachelor-Project
```

### Running the Models
Navigate to the `notebooks/` directory and open the desired Jupyter notebook to train and evaluate models. Or, use scripts:


## Conclusion
}All three models showed strong capabilities in detecting curbs. In the test data validation, the LSTM Autoencoder with a low value threshold delivered the best overall performance, achieving a precision of 0.493, recall of 1.0, and an F1-score of 0.660. However, its performance during real-world validation was somewhat disappointing, with a precision of 0.70, recall of 0.75, and F1-score of 0.72. The Autoencoder with a low value threshold ranked in the middle, with test validation results of 0.384 precision, 1.0 recall, and an F1-score of 0.555. In real-world validation, it achieved a precision of 0.56, recall of 0.83, and F1-score of 0.67. The One-Class SVM consistently exhibited the weakest performance, scoring a precision of 0.33, recall of 1.0, and F1-score of 0.50 in test validation, and a precision of 0.49, recall of 0.83, and F1-score of 0.61 in the real world validation.

High value of false positive of all three methods because of our goal is to detect as many anomalies as possible. While the LSTM Autoencoder excelled in the controlled testing environment, its reduced effectiveness in real world scenarios can be attributed to factors such as the lack of overlapping windowing during deployment, limited computational resources, and challenges in data labeling. The labeling process, in particular, introduced notable inconsistencies, as human generated annotations tend to carry inherent inaccuracies. Improving label quality and consistency could significantly enhance validation results. 

## Future Work
It is essential to develop algorithms that effectively handle labeling inaccuracies. Additionally, incorporating additional sensor data, such as accelerations along other axes and gyroscope measurements, optimizing the LSTM architecture, can further enhance the precision and reliability of the model. Thirdly, evaluating sensor placements at various positions on the bicycle could substantially improve the model's practicality.

## Acknowledgments

- **Author**: Ziyi Liu  
- **Supervisor**: Dandan Liu, M.Sc., University of Kassel  
- **Institute**: Communication Technology, University of Kassel, Germany  

This project was completed as part of the Bachelor's degree requirements at the University of Kassel. I would like to express my gratitude to my supervisor, Dandan Liu, for her valuable guidance and support throughout this project.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


