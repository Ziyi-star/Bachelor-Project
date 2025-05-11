# Bachelor Project: Anomaly Detection Techniques for Cyclist Curb Recognition using Traditional Machine Learning und Deep Learning Techniques for Time Series Classification

## Overview

This repository contains the code and documentation for my Bachelor's project, focusing on the application of deep learning techniques to time series classification. The objective is to explore and evaluate one class SVM, Autoencoder and LSTM Autoencoder architectures to classify in time-series data.

## General Information

- **Author**: Ziyi Liu  
- **Supervisor**: Dandan Liu, M.Sc., University of Kassel  
- **Institute**: Communication Technology, University of Kassel, Germany  

This repository was created as part of the bachelor project *Anomaly Detection Techniques for Cyclist Curb Recognition*.

## Project Structure

- `data/` – Contains datasets used for training and evaluation, including preprocessed time series data.  
- `models/` – Different Models Autoencoder, LSTM Autoencoder, one class SVM
- `notebooks/` – Jupyter notebooks for exploratory data analysis, model training, and evaluation.
- `results/` – Evaluation Results, Plots and other visualizations  
- `scripts/` – Python scripts  
- `utils/` – Utility functions for this project  
- `LICENSE` – Information about the terms of use for this repository  


## Key Features

- **Data Preprocessing**: Handling missing values, normalization, and windowing of time series data.
- **Model Implementations**:
  - One Class SVM
  - Autoencoder
  - Long Short-Term Memory (LSTM)

- **Training Pipeline**: Modular training scripts with support for early stopping, learning rate scheduling, and checkpointing.
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
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Models
Navigate to the `notebooks/` directory and open the desired Jupyter notebook to train and evaluate models. Or, use scripts:



## Conclusion


## Future Work



## Acknowledgments

This project was completed as part of the Bachelor's degree requirements at University Kassel. Thanks to my supervisor Dandan Liu for guidance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


