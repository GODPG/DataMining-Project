# Server Machine Dataset (SMD) - Anomaly Detection using ConvBiGRU Autoencoder

## Data Set Information:

The Server Machine Dataset (SMD) is a publicly available dataset designed for anomaly detection in server machine telemetry data. Collected by NetMan and derived from the research paper "Robust Anomaly Detection for Multivariate Time Series through Stochastic RNN," this project uses a ConvBiGRU Autoencoder model to detect anomalies in this dataset.

#### Key Dataset Characteristics:

*   **Source:** Collected and made available by NetMan (see [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly))
*   **Related Publication:** "Robust Anomaly Detection for Multivariate Time Series through Stochastic RNN"
*   **Time Span:** 5 weeks of data
*   **Granularity:** 1-minute intervals (time index omitted from data files)
*   **Number of Entities:** 28 distinct server machines (entities)
*   **Dimensionality:** 38 data dimensions (metrics) per entity
*   **Dataset Size:** Approximately 5,724,602 data points
*   **Train/Test Split:** Approximately 1:1
*   **Labels:** Training set has no anomaly labels; testing set includes anomaly labels.
*   **Interpretation Labels:** Provides lists of dimensions contributing to each anomaly (identifies the specific metrics causing the anomaly).

#### Business Understanding

##### What is Anomaly Detection in Server Machines?

Modern data centers and cloud infrastructure rely on the stable operation of thousands of servers. Anomaly detection plays a crucial role in identifying unusual server behavior that may indicate underlying issues, such as hardware failures, software bugs, or security breaches. Early detection of anomalies can prevent service disruptions, reduce downtime, and improve overall system reliability.

##### Why ConvBiGRU Autoencoder for Anomaly Detection?

This project leverages a ConvBiGRU Autoencoder for anomaly detection due to its ability to:

*   **Capture Temporal Dependencies:** The BiGRU (Bidirectional Gated Recurrent Unit) component excels at capturing temporal dependencies in time-series data, allowing the model to learn the normal sequential patterns of server behavior.
*   **Extract Spatial Features:** The Convolutional (Conv) layers extract spatial features from the telemetry data, identifying patterns and relationships among different metrics.
*   **Learn Compressed Representation:** The Autoencoder architecture learns a compressed representation of the normal data, enabling it to detect anomalies as deviations from this learned representation.
*   **Robust Anomaly Detection:** By combining convolutional and recurrent layers within an autoencoder framework, this model achieves robust anomaly detection performance compared to traditional methods.

##### Using Interpretation Labels

The interpretation_label.txt provides valuable insights into the cause of anomalies. The ability to automatically determine which metric is driving the anomaly is very useful.

#### Source:

[https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)

## Data Understanding


#### Attribute Information:

The SMD dataset consists of multiple text files, each representing the telemetry data from a specific server machine. The files are named in the format `machine-x-y.txt`, where `x` represents the group, and `y` is the index within the group.

Each line in the file represents a 1-minute interval and contains 38 comma-separated values representing the server's metrics. Specific metrics are not named but represent key performance indicators of the server.

The dataset includes the following files:

*   **train/:** Directory containing training data files (`machine-x-y.txt`). Training data includes only normal behavior.
*   **test/:** Directory containing testing data files (`machine-x-y.txt`). Testing data contains a mix of normal and anomalous behavior.
*   **test_label.txt:** Contains the anomaly labels for the test set. Each line corresponds to a data point in the test set and indicates whether it is normal (0) or anomalous (1).
*   **interpretation_label.txt:** Provides lists of dimensions that contribute to each anomaly in the test set. This file helps to identify which specific metrics are responsible for the detected anomalies.

## Team Collaboration - Directory Structure

*Adjust based on your actual project structure.*


#### Instructions

1.  Download the SMD dataset from [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly).
2.  Create the directory structure as shown above.
3.  Place the data files in the `data` directory, ensuring the correct folder structure for `train` and `test` data.
4.  Run the notebooks in sequence to perform data analysis, data preprocessing, model building, and anomaly detection evaluation.
5.  The trained ConvBiGRU Autoencoder model will be saved in the `models` directory.

#### Suggested data pre-processing steps
1.  **Data Windowing:** Since there is no time index, it's important to create sequences, using sliding window.
2.  **Reshape Data:** Reshape the data into the format [Samples, timesteps, features]
3.  **Scaling data:** Use a scaler, like MinMaxScaler to scale the data between 0 and 1.

## Data Preparation and Visualization

*Adapt based on your specific implementation.*


*   **Data Loading:** Load the data from the `.txt` files into Pandas DataFrames.
*   **Data Scaling:** Standardize or normalize the telemetry data to ensure optimal model performance.
*   **Data Windowing:** Divide the time-series data into overlapping or non-overlapping windows for sequence-based learning.
*   **Label Alignment:** Ensure that the labels in `test_label.txt` are correctly aligned with the corresponding data points in the test set.

## Data Cleansing, Processing, and Modeling

*Adapt based on your specific implementation.*


*   **ConvBiGRU Autoencoder Architecture:**
    *   **Encoder:** Consists of Convolutional layers for feature extraction, followed by Bidirectional GRU layers to capture temporal dependencies. The encoder compresses the input data into a lower-dimensional latent space.
    *   **Decoder:** Reconstructs the original input data from the latent space representation using GRU and Deconvolutional layers.
*   **Loss Function:** Mean Squared Error (MSE) is commonly used to measure the reconstruction error between the input and output data.
*   **Anomaly Detection Threshold:** Anomaly scores are calculated based on the reconstruction error. A threshold is set to classify data points with high reconstruction errors as anomalies.
*   **Training with Interpretation Labels:** By using the `interpretation_label.txt` file, the model can be designed to take in account, which of the 38 metrics are most responsible for the specific anomaly.

**Key Implementation Details:**

*   **Hyperparameter Tuning:** Experiment with different hyperparameters, such as the number of convolutional layers, GRU units, learning rate, batch size, and window size, to optimize the model's performance.
*   **Early Stopping:** Implement early stopping to prevent overfitting and select the best model based on a validation set.
*   **Metrics:** Common metrics are, Precision, Recall, F1 Score.

## Process Summary

By performing ConvBiGRU Autoencoder model, we aimed to get better at predictively detecting following
*   Minimize downtime
*   Proactive remediation of issues
*   Improve overall system reliability
*   Reduce operational costs associated with reactive incident response.
*   Optimize resource allocation based on predicted system needs.
*   Improved anomaly detection accuracy compared to traditional methods
*   Identifying key metrics contributing to anomalies to enable targeted remediation efforts

## Additional Information
1.  **Interpreting the anomaly results:** The anomaly results can be further interpreted by referring back to the interpretation_label.txt file, to find out, which of the 38 metrics are responsible.


