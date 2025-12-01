# A Machine Learning Approach to Bitcoin Support–Resistance Trading Signals

This project builds a machine learning pipeline to identify **Bitcoin trading signals** (“long” or “short”) using **technical analysis**, with a focus on **support and resistance levels**.

We combine:

- High-frequency BTC/USD price data  
- Gold (XAU) data as a macro/hedge asset  
- Technical indicators and support–resistance zones  
- Supervised learning models (XGBoost, LSTM)  
- A **triple-barrier labeling** framework for defining “good” long/short opportunities  

The goal is to move from **emotion-driven trading** to **data-driven, rules-based decisions**.

---

## Table of Contents

1. [Project Motivation](#project-motivation)  
2. [Research Question](#research-question)  
3. [Core Concepts & ML Principles](#core-concepts--ml-principles)  
4. [Results (High-Level)](#results-high-level)  
5. [Repository Structure](#repository-structure)  

---

## Project Motivation

Human traders often make decisions under **fear, greed, and cognitive biases**:

- **Prospect theory** shows that people fear losses more than they value equivalent gains (loss aversion).  
- Biases like **overconfidence**, **herd behavior**, and **recency bias** can lead to irrational trades.

Support and resistance levels are widely used in technical analysis, but are usually:

- Drawn **by eye**,  
- Subject to **interpretation** and **emotions**.

**Why this project?**

- Use **machine learning** and **algorithmic rules** to identify and test support–resistance based strategies.
- Replace “gut feeling” with **data-driven, repeatable rules** for when to go **long** or **short** Bitcoin.

---

## Research Question

> **How can we develop a set of technical analysis rules by algorithmically identifying and analyzing the most reliable support and resistance levels in historical BTC price data, to guide an investor’s decision to go “long” or “short”?**

---

## Core Concepts & ML Principles

### Technical Analysis & Support–Resistance

- **Technical Analysis**: Evaluating price and volume data instead of fundamentals.
- **Support Level**: A price area where downward moves tend to pause or reverse.
- **Resistance Level**: A price area where upward moves tend to pause or reverse.

In the code, we:

- Compute **rolling support/resistance** from BTC prices.
- Cluster them with **K-Means** to find **zones** instead of single lines.
- Measure how far the current price is from these zones.

### Behavioral Finance & Prospect Theory

- Prospect theory explains why traders:
  - Hate losses more than they like gains (loss aversion).
  - May hold losing positions too long or close winners too early.
- By using **algorithmic labels** (triple-barrier method), we try to define **rational** long/short opportunities instead of emotional ones.

### Machine Learning Principles Used

1. **Supervised Learning (Binary Classification)**
   - Target: whether a short-term trade hits a **take-profit (1)** or **stop-loss (0)** first.
   - Models:
     - **XGBoost** (gradient boosted trees).
     - **LSTM** (sequence model using PyTorch) to capture time dynamics.

2. **Time Series Modeling**
   - Features built from **historical BTC and Gold prices**.
   - Uses **lagged values**, **rolling indicators**, and **sequences of 30 minutes** for the LSTM.

3. **Feature Engineering**
   - Technical indicators:
     - SMA, EMA, TEMA  
     - MACD  
     - RSI  
     - Bollinger Bands (upper, lower, width)  
     - ATR  
     - VWAP  
     - Volatility (rolling std), range (high – low)
   - BTC–Gold combined features:
     - BTC OHLCV + Gold OHLCV at the **same timestamps**.
   - Support/resistance distance features:
     - `dist_to_support`, `dist_to_resistance`.

4. **Triple-Barrier Labeling**
   - For each point in time:
     - Take-profit barrier at +0.3%  
     - Stop-loss barrier at −0.3%  
     - Time barrier at 30 minutes
   - Label = which barrier is hit first:
     - **1** → Take-profit hit first (favorable move)
     - **0** → Stop-loss hit first (unfavorable move)

5. **Imbalanced Data Handling**
   - More “take-profit” events than “stop-loss” → class imbalance.
   - XGBoost: uses `scale_pos_weight`.
   - LSTM: uses **class-weighted cross-entropy**.

6. **Monte Carlo Simulation (Risk/Scenario Analysis)**
   - Uses a simple **Geometric Brownian Motion** model to simulate many future BTC paths.
   - Derives **probabilistic support/resistance bands** (5th & 95th percentiles).
  
## Results (High-Level)

### **XGBoost Classifier**
- **Accuracy:** ~70% on the test set  
- Performs significantly better on the majority class (**take-profit** events)  
- Demonstrates strong baseline performance using engineered features  

### **LSTM Model**
- **Accuracy:** ~50% in the current configuration  
- Shows a different performance trade-off:
  - Better recall for **stop-loss** (minority) class  
  - Weaker performance on take-profit class without further tuning  

### **What the Results Show**
- Technical indicators + support/resistance features **do contain useful predictive signals**  
- Models can identify short-term directional patterns in BTC price movements  
- These are **baseline results**, not full trading strategies  
- There is substantial room for improvement through:
  - Hyperparameter tuning  
  - Additional feature engineering  
  - Market-regime detection  
  - More advanced sequence models  



---

## Repository Structure

```text
Ignite_AI4ALL_final_project/
├── data/
│   ├── data.csv              # Raw BTC/USD 1-minute data
│   ├── XAU_1m_data.csv       # Raw Gold (XAU) 1-minute data
│   └── btc_final_data.csv    # Engineered dataset (created by dataset_creation.ipynb)
├── dataset_creation.ipynb    # Data cleaning & feature engineering
├── prediction.ipynb          # Clustering, Monte Carlo, triple-barrier, XGBoost, LSTM
└── README.md                 # This file
