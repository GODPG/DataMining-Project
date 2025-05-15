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

