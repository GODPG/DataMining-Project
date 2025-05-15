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
Missing Values? None
</pre>

#### Attribute Information:

The SMD dataset consists of text files in `machine-x-y.txt` format, with each line representing a 1-minute interval and containing 38 server metrics.

*   **train/:** Training data (normal behavior only).
*   **test/:** Testing data (normal and anomalous behavior).
*   **test_label.txt:** Anomaly labels (0 or 1).
*   **interpretation_label.txt:** Metrics contributing to each anomaly.

## Team collaboration - directory structure
PGspeed:
[文件]

PGspeed:
#### Instructions
<pre>
- Clone the GitHub repository
- Run the notebooks in sequence
- Trained ConvBiGRU Autoencoder model will be in the models directory.

├── Processed_data
│   ├── machine-1-1/
│   │   ├── machine-1-1_test.csv
│   │   └──  machine-1-1_train.csv
├── SeverMachineDataset
│   ├── interpretation_label
│   │   └── machine-1-1.txt
│   ├── test
│   │   └── machine-1-1.txt
│   ├── test_label
│   │   └── machine-1-1.txt
│   ├── train
│   │   └── machine-1-1.txt
│   └── LICENSE
├── code
│   ├── data_cleaning.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   └── visualize.py
├── saved_models
│   ├── conv_bi_gru_autoencoder.pth
│   └── scaler_new.pkl
├── README.md
├── Technical Report.ipynb
└── requirement.txt
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
Packages: Pandas, NumPy, Matplotlib, Seaborn, plotly
</pre>

## Data Cleansing, processing, and modeling
<pre>
Code Used: Python
Packages: Pandas, scikit-learn, torch, torch.nn, torch.optim
</pre>

**ConvBiGRU Autoencoder Model:**
<pre>
class ConvBiGRUAutoencoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(inp_dim, inp_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(inp_dim * 2),
            nn.Conv1d(inp_dim * 2, inp_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(inp_dim * 2)
        )
        self.enc = nn.GRU(
            inp_dim * 2,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dec = nn.GRU(
            hid_dim * 2,
            inp_dim,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        B = x.size(0)
        h0 = torch.zeros(
            self.enc.num_layers * 2,
            B,
            self.enc.hidden_size,
            device=x.device
        )
        enc_out, _ = self.enc(x, h0)
        dec_out, _ = self.dec(enc_out)
        return dec_out
</pre>

**Training Process:**
<pre>
1.  Load data using Data_Cleaning.py, including features, standardizing, and sliding window processing.
2.  If the existing model & scaler exists, directly load it.
3.  Otherwise, train a new model: define hyperparameters such as learning rate, batch size, and number of epochs.
4.  Define the ConvBiGRUAutoencoder model structure and move it to the GPU or CPU for training.
5.  Calculate static weights based on anomaly logs to enhance the model's attention to key metrics.
6.  Use the Adam optimizer and CosineAnnealingLR learning rate scheduler.
7.  Use MSE loss function and train in a loop. 
8.  Implement early stopping to save the best model.
</pre>

**Key Hyperparameters:**
<pre>
- seq_length = 20
- hidden_size = 64
- num_layers = 2
- dropout = 0.2
- lr = 1e-3
- batch_size = 64
- num_epochs = 50
- alpha = 1.0
</pre>

**Evaluation:**
<pre>
1.  The test data is split into validation and test sets.
2.  Search for the best threshold on the validation set to maximize F1 score.
3.  Evaluate the final performance of the model on the test set.
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
