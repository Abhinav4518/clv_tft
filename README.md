# 📈 Temporal Fusion Transformer: Dynamic CLV & Forecaster

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14+-000000?style=for-the-badge&logo=next.js&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg?style=for-the-badge)

An end-to-end, full-stack machine learning architecture designed to forecast Customer Lifetime Value (CLV) using a custom-built **Temporal Fusion Transformer (TFT)**. 

Unlike traditional autoregressive models or vanilla LSTMs, this architecture effectively processes mixed-type time-series data (static metadata, known future inputs, and volatile past observations) to generate highly interpretable, asymmetric prediction intervals.

## 🧠 Core Architecture Highlights
* **Custom TFT Implementation:** Built from scratch using `tf.keras.Model` subclassing. Features custom Gated Residual Networks (GRNs) and Variable Selection Networks (VSNs).
* **Quantile Loss Optimization:** Generates an 80% confidence interval ($P_{10}$, $P_{50}$, $P_{90}$) instead of standard deterministic point predictions.
* **Dynamic Time Machine API:** The FastAPI backend dynamically slices multi-dimensional tensors based on user-selected anchor dates, allowing for historical backtesting and flexible forecasting.
* **Modern Web Dashboard:** A Next.js (TypeScript/Tailwind) frontend utilizing Recharts to render the probabilistic forecast funnel in real-time.

## 🏗️ Project Structure
```text
clv_tft_forecaster/
├── api/                  # FastAPI backend and Docker config
├── frontend/             # Next.js React web application
├── src/                  # Core TensorFlow layer definitions
├── notebooks/            # Exploratory Data Analysis (EDA)
├── train.py              # ML Training loop with tf.data.AUTOTUNE
└── generate_data.py      # Synthetic time-series data generator