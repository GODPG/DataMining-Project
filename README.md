<center>
    <center>
        <img src = "images/convbigru.png" width = 35%/>
    </center>
</center>

## Data Set Information:

The SMD (Server Machine Dataset) is designed for anomaly detection in server machine telemetry data collected by NetMan. Derived from the "Robust Anomaly Detection for Multivariate Time Series through Stochastic RNN" paper, this project uses a ConvBiGRU Autoencoder for anomaly detection.

#### Business Understanding

##### What is Anomaly Detection in Server Machines?

Modern data centers rely on thousands of servers. Anomaly detection identifies unusual server behavior that may indicate hardware failures or security breaches. Early detection prevents service disruptions.

##### Why ConvBiGRU Autoencoder?

The ConvBiGRU Autoencoder captures temporal dependencies and spatial features in the telemetry data. Its unsupervised nature allows it to learn normal server behavior and detect deviations.

*   **Temporal Dependencies:** BiGRU captures temporal patterns.
*   **Spatial Features:** Convolutional layers extract spatial features from metrics.
*   **Unsupervised Learning:** Learns normal data representations without labeled anomalies.

##### Using Interpretation Labels

The `interpretation_label.txt` identifies the metrics contributing to each anomaly, facilitating root cause analysis.

#### Source:

NetMan ([https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly))

## Data Understanding
<pre>
Data Set Characteristics:  Time-Series, Multivariate
Area: System Monitoring, Anomaly Detection
Attribute Characteristics: Real
Missing Values? [Specify if there are missing values, and how they are handled]
</pre>

#### Attribute Information:

The SMD dataset consists of text files in `machine-x-y.txt` format, with each line representing a 1-minute interval and containing 38 server metrics.

*   **train/:** Training data (normal behavior only).
*   **test/:** Testing data (normal and anomalous behavior).
*   **test_label.txt:** Anomaly labels (0 or 1).
*   **interpretation_label.txt:** Metrics contributing to each anomaly.

## Team collaboration - directory structure

#### Instructions
<pre>
- Clone the GitHub repository
- Run the notebooks in sequence
- Trained ConvBiGRU Autoencoder model will be in the models directory.

├── data
│   ├── train/
│   │   ├── machine-1-1.txt
│   │   ├── machine-1-2.txt
│   │   └── ...
│   ├── test/
│   │   ├── machine-1-1.txt
│   │   ├── machine-1-2.txt
│   │   └── ...
│   ├── test_label.txt
│   ├── interpretation_label.txt
├── models
│   ├── convbigru_autoencoder.h5
│   ├── convbigru_autoencoder.json
├── notebooks
│   ├── 1. exploratory_data_analysis.ipynb
│   ├── 2. data_preprocessing.ipynb
│   ├── 3. model_building_training.ipynb
│   ├── 4. anomaly_detection_evaluation.ipynb
├── README.md
</pre>

#### Data Pre-processing Steps
<pre>
1.  Data Windowing: Create sequences using a sliding window.
2.  Reshape Data: [Samples, timesteps, features]
3.  Scaling data: Use MinMaxScaler to scale between 0 and 1.
</pre>

## Data Preparation and Visualization
<pre>
Code Used: Python
Packages: Pandas, NumPy, Matplotlib, Seaborn
</pre>

## Data Cleansing, processing, and modeling
<pre>
Code Used: Python
Packages: Pandas, scikit-learn, TensorFlow, Keras
</pre>

**ConvBiGRU Autoencoder Architecture:**
<pre>
- Encoder: Convolutional layers + Bidirectional GRU layers
- Decoder: GRU layers + Deconvolutional layers
- Loss Function: Mean Squared Error (MSE)
- Anomaly Detection Threshold: Reconstruction error based threshold
</pre>

## Process Summary
**By performing ConvBiGRU Autoencoder model, we aimed to get better at predictively optimizing following**
- Minimize downtime
- Proactive remediation of issues
- Improve overall system reliability
- Reduce operational costs
- Optimize resource allocation
- Improved anomaly detection accuracy
- Identifying key metrics contributing to anomalies

<center>
    <img src = "images/copyright.png" width = 15%, align = "right"/>
</center>
