Variational Autoencoder (VAE) for Anomaly Detection

This project implements a Variational Autoencoder (VAE) from scratch using PyTorch for unsupervised anomaly detection on a high-dimensional dataset. The model leverages both reconstruction error and latent space divergence (KL divergence) to identify anomalous samples and compares performance against a classical baseline (Isolation Forest).

📌 Project Overview

Anomaly detection is critical in domains such as:

Network intrusion detection

Fraud detection

Sensor fault diagnosis

This project demonstrates how probabilistic latent variable models, specifically VAEs, can effectively learn the underlying distribution of normal data and flag deviations as anomalies.

🚀 Key Features

Custom VAE implementation (Encoder, Decoder, Reparameterization Trick)

β-VAE with KL Annealing to stabilize training

High-dimensional synthetic dataset with injected anomalies

Robust anomaly scoring mechanism

Quantitative evaluation using Precision, Recall, F1-Score, ROC AUC

Performance comparison with Isolation Forest

🗂 Project Structure
.
├── vae.py        # Main Python implementation
└── README.md     # Project documentation

📊 Dataset Description

The dataset is programmatically generated using sklearn.make_blobs:

Total samples: 10,000

Features: 20

Normal data: Clustered Gaussian blobs

Anomalies: Uniformly distributed outliers

Anomaly ratio: 5%

All features are standardized using StandardScaler.

🏗 Model Architecture
Encoder

Input → 128 → 64

Outputs:

Mean vector (μ)

Log-variance vector (log σ²)

Latent Space

Latent dimension: 8

Sampling via reparameterization trick

Decoder

Latent → 64 → 128 → Output

🧮 Loss Function

The VAE optimizes the following objective:

𝐿
=
Reconstruction Loss
+
𝛽
⋅
KL Divergence
L=Reconstruction Loss+β⋅KL Divergence

Reconstruction Loss: Mean Squared Error (MSE)

KL Divergence: Regularizes latent space to follow a standard normal distribution

β (Beta): Controls strength of latent regularization

🔧 Optimization Strategy

Optimizer: Adam (lr = 1e-3)

Batch size: 128

Epochs: 50

KL Annealing:
Gradually increases β from 0 → 1 over the first 20 epochs to prevent posterior collapse.

🚨 Anomaly Detection Method

An anomaly score is computed for each sample as:

Anomaly Score
=
Reconstruction Error
+
KL Divergence
Anomaly Score=Reconstruction Error+KL Divergence

Samples above the 95th percentile of anomaly scores are classified as anomalies.

📈 Evaluation Metrics

Precision

Recall

F1-Score

ROC AUC

Evaluation is performed on a held-out test set with known anomalies.

📊 Baseline Comparison

A scikit-learn Isolation Forest model is used as a baseline for comparison.

Model	Strengths	Weaknesses
VAE (β-VAE)	Learns data distribution, probabilistic	Requires tuning
Isolation Forest	Fast, simple	Less effective in high dimensions

The VAE consistently achieves higher F1-Score and ROC AUC due to its ability to model complex data distributions.

▶️ How to Run
Install Dependencies
pip install numpy torch scikit-learn matplotlib

Run the Project
python vae.py


The script will:

Generate the dataset

Train the VAE

Detect anomalies

Evaluate performance

Compare with Isolation Forest

 Key Learning Outcomes

Understanding of Variational Autoencoders

Practical implementation of KL divergence & reparameterization

Importance of latent space regularization

Application of deep generative models for anomaly detection

Performance comparison with classical ML methods

📌 Future Improvements

Apply to real datasets (e.g., KDD Cup 1999)

Visualize latent space using t-SNE or PCA

Adaptive threshold selection via ROC curve

Convolutional VAE for structured data

📄 License

This project is intended for academic and educational use.



