# Server Machine Dataset (SMD) - 使用 ConvBiGRU 自编码器进行异常检测

## Data Set Information (数据集信息):

The Server Machine Dataset (SMD) is a publicly available dataset designed for anomaly detection in server machine telemetry data. Collected by NetMan and derived from the research paper "Robust Anomaly Detection for Multivariate Time Series through Stochastic RNN," this project uses a ConvBiGRU Autoencoder model to detect anomalies in this dataset.

SMD（服务器机器数据集）是一个公开数据集，专为服务器机器遥测数据中的异常检测而设计。由NetMan收集，源于论文 "Robust Anomaly Detection for Multivariate Time Series through Stochastic RNN"，本项目使用 ConvBiGRU 自编码器模型来检测该数据集中的异常。

#### Key Dataset Characteristics (数据集关键特征):

*   **Source (来源):** Collected and made available by NetMan (see [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)) (由NetMan收集并提供 (参见 [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)))
*   **Related Publication (相关论文):** "Robust Anomaly Detection for Multivariate Time Series through Stochastic RNN" ("基于随机RNN的多变量时间序列的鲁棒异常检测")
*   **Time Span (时间跨度):** 5 weeks of data (5周数据)
*   **Granularity (时间粒度):** 1-minute intervals (time index omitted from data files) (1分钟间隔 (数据文件中省略时间索引))
*   **Number of Entities (实体数量):** 28 distinct server machines (entities) (28个不同的服务器机器 (实体))
*   **Dimensionality (维度):** 38 data dimensions (metrics) per entity (每个实体38个数据维度 (指标))
*   **Dataset Size (数据集大小):** 5,724,602 data points (大约) (5,724,602 个数据点 (大约))
*   **Train/Test Split (训练/测试集划分):** Approximately 1:1 (大约 1:1)
*   **Labels (标签):** Training set has no anomaly labels; testing set includes anomaly labels. (训练集没有异常标签; 测试集包含异常标签)
*   **Interpretation Labels (解释标签):** Provides lists of dimensions contributing to each anomaly (identifies the specific metrics causing the anomaly). (提供导致每个异常的维度列表 (识别导致异常的具体指标))

#### Business Understanding (业务理解)

##### What is Anomaly Detection in Server Machines? (服务器机器中的异常检测是什么？)

Modern data centers and cloud infrastructure rely on the stable operation of thousands of servers. Anomaly detection plays a crucial role in identifying unusual server behavior that may indicate underlying issues, such as hardware failures, software bugs, or security breaches. Early detection of anomalies can prevent service disruptions, reduce downtime, and improve overall system reliability.

现代数据中心和云基础设施依赖于成千上万台服务器的稳定运行。异常检测在识别异常服务器行为方面起着至关重要的作用，这些行为可能表明潜在的问题，例如硬件故障、软件错误或安全漏洞。提前检测到异常可以防止服务中断、减少停机时间并提高整体系统可靠性。

##### Why ConvBiGRU Autoencoder for Anomaly Detection? (为什么使用 ConvBiGRU 自编码器进行异常检测？)

This project leverages a ConvBiGRU Autoencoder for anomaly detection due to its ability to:

*   **Capture Temporal Dependencies (捕捉时间依赖性):** The BiGRU (Bidirectional Gated Recurrent Unit) component excels at capturing temporal dependencies in time-series data, allowing the model to learn the normal sequential patterns of server behavior. (BiGRU (双向门控循环单元) 组件擅长捕捉时间序列数据中的时间依赖性，使模型能够学习服务器行为的正常顺序模式)
*   **Extract Spatial Features (提取空间特征):** The Convolutional (Conv) layers extract spatial features from the telemetry data, identifying patterns and relationships among different metrics. (卷积 (Conv) 层从遥测数据中提取空间特征，识别不同指标之间的模式和关系)
*   **Learn Compressed Representation (学习压缩表示):** The Autoencoder architecture learns a compressed representation of the normal data, enabling it to detect anomalies as deviations from this learned representation. (自编码器架构学习正常数据的压缩表示，使其能够将异常检测为与这种学习表示的偏差)
*   **Robust Anomaly Detection (鲁棒的异常检测):** By combining convolutional and recurrent layers within an autoencoder framework, this model achieves robust anomaly detection performance compared to traditional methods. (通过在自编码器框架内组合卷积层和循环层，该模型与传统方法相比实现了鲁棒的异常检测性能)

##### Using Interpretation Labels (使用解释性标签)

The interpretation_label.txt provides valuable insights into the cause of anomalies. The ability to automatically determine which metric is driving the anomaly is very useful.

interpretation_label.txt文件为异常的原因提供了有价值的见解。 自动确定哪个指标驱动异常的能力非常有用。

#### Source (数据来源):

[https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)

## Data Understanding (数据理解)

