# ConvBiGRU Autoencoder Anomaly Detection Project

## Introduction

This project aims to develop a machine learning-based anomaly detection system for identifying unusual behavior in time-series data. We leverage a ConvBiGRU Autoencoder model to learn patterns from normal data and detect anomalies by identifying deviations from those patterns.

## Problem Statement

In many real-world applications, such as server monitoring and industrial equipment maintenance, the timely detection of anomalous behavior is critical. This project addresses the following questions:

*   How can we accurately detect anomalous behavior in time-series data?
*   How can we utilize machine learning techniques to identify normal and abnormal patterns?
*   How can we build an easily understandable and deployable anomaly detection system?

## Key Findings

*   We successfully developed an anomaly detection model based on a ConvBiGRU Autoencoder.
*   The model can effectively learn patterns from normal data and identify anomalies by detecting deviations from those patterns.
*   Experimental results demonstrate that the model performs well on the test dataset.
*   The use of Winsorize clipping significantly improved the model's performance and robustness.

## Model Performance

The model achieved the following performance metrics on the test dataset:

*   **Accuracy:** \[0.940]
*   **F1-score:** \[0.680]
*   **AUC:** \[0.965]
*   **Best Threshold:** \[13.341750]

Confusion Matrix:
[12475  332]
[519    904]


These results indicate that the model can identify anomalous behavior with reasonable accuracy.

## Potential Improvements

Here are some potential areas for improvement:

*   **Hyperparameter Optimization:** Further tuning the model's hyperparameters (e.g., learning rate, batch size, number of layers) could improve performance.
*   **Feature Engineering:** Adding more relevant features, such as statistical or domain-specific features, could enhance the model.
*   **Ensemble Learning:** Experimenting with ensemble methods (combining multiple models) could improve robustness and accuracy.
*   **Model Interpretability:** Investigating ways to improve the model's interpretability to better understand its decision-making process.

## Usage Instructions

1.  **Environment Setup:** Ensure that you have installed all necessary libraries, including `torch`, `scikit-learn`, `pandas`, `plotly`, etc.
2.  **Data Preparation:** Place your training and testing data in the correct location.
3.  **Run the Script:** Execute the `main.py` script to train, evaluate, and visualize the model.

## Contributions

Contributions of any kind are welcome, including code, documentation, and suggestions.

## License

\[Insert your license information here]

