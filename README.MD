# Resilient Traffic Forecasting Under Extreme Data Degradation: A Conditional Sequence GAN Approach

## Project Overview
Intelligent Transportation Systems (ITS) rely heavily on continuous sensor data to predict traffic flow and manage city gridlock. However, physical sensors frequently fail due to hardware malfunctions, network latency, or power outages. 

This project explores the limits of deep learning forecasting models under extreme missing data conditions (up to 80% degradation) on the Seattle highway traffic dataset. We demonstrate that traditional models fail under these conditions by falling into the "Mean-Prediction Trap." To solve this, we propose a lightweight, highly efficient **Conditional Sequence GAN** (using LSTM and Conv1D architectures) that explicitly preserves the physical volatility of traffic shockwaves, bypassing the limitations of purely MSE-driven models.

---

## The Research Gaps

Through replicating state-of-the-art baseline models, we identified three critical flaws in current time-series forecasting literature:

1. **The Mean-Prediction Trap:** Traditional models are optimized purely on Mean Squared Error (MSE). When faced with high uncertainty (80% missing data), they minimize their mathematical penalty by predicting flat, safe statistical averages. They mathematically refuse to predict the sharp, volatile drops of real traffic crashes.
2. **Cascading Pipeline Errors:** Standard approaches use two-step pipelines: first imputing missing data, then forecasting. If the imputation algorithm guesses wrong, the forecaster is permanently doomed.
3. **Extreme Degradation Benchmarks:** Most research benchmarks on 10% to 20% missing data. Real-world physical failures cause massive, continuous block-missingness. We stress-tested our models up to 80% degradation.

---

## Our Methodology & Architecture

To address these gaps, we engineered a **Conditional Sequence GAN** featuring an End-to-End Joint Training pipeline.

* **The Generator (LSTM):** Condenses 323 physical sensors into a 64-dimensional hidden state, processing a 12-step history sequence to output a 1-step, 5-minute forecast. By predicting directly from corrupted data, it acts as an implicit imputer, eliminating cascading pipeline errors.
* **The Discriminator (Conv1D -> LSTM):** Evaluates the concatenated 13-step sequence. The Conv1D layer acts as a spatial edge detector to find traffic shockwaves across the 323 sensors, while the LSTM tracks temporal continuity. 
* **The Hybrid Loss Function:** The Generator optimizes against a combined objective to ensure both physical realism (via adversarial loss) and chronological accuracy (via a scaled physical anchor):
  
  $$\mathcal{L}_G = -\log(D(X \oplus \hat{Y})) + \lambda \text{MSE}(\hat{Y}, Y)$$

By operating at just **216,000 parameters** (compared to ~3.3 million in baseline BiLSTM models), our architecture is uniquely positioned for lightweight, real-time edge deployment.

---

## Repository Structure

This repository contains the Phase 1 experimentation, organized into three core files:

1. `replication_paper.ipynb` *(or .py)*: Contains the implementation and evaluation of the baseline state-of-the-art model. This establishes the benchmark and explicitly demonstrates the model falling into the Mean-Prediction Trap at high degradation.
2. `25_combination_run.ipynb` *(or .py)*: The architectural grid search. We trained and evaluated 25 different GAN configurations to find the optimal balance of depth, width, and adversarial stability without unnecessarily ballooning the parameter count.
3. `final_model_results.ipynb` *(or .py)*: The execution and evaluation of our Champion GAN model, comparing its temporal variance, spatial accuracy, and error metrics against the baseline.

---

##  Key Results & Visualizations

While the baseline model achieved a slightly lower overall Mean Absolute Error (MAE), our research proves that MAE is a deceptive metric in anomaly forecasting. The baseline achieves its score by artificially erasing up to 16% of the highway's physical variance. 

Our GAN successfully forces the prediction to maintain the dataset's natural variance, capturing the sharp drops of extreme traffic events that the baseline completely ignores.

### 1. Architectural Grid Search
We evaluated 25 GAN combinations to locate the optimal feature extraction boundaries before recurrent memory starvation occurs. 

![Grid Search Heatmap](path/to/your/heatmap_image.png)
*(Replace `path/to/your/heatmap_image.png` with your actual file path. This heatmap shows the validation performance across the different depth/width configurations.)*

### 2. The Mean-Prediction Trap (Micro-Zoom)
When viewing a macro 24-hour cycle, predictions look similar. However, isolating a severe micro-event (a rapid speed drop) reveals the baseline's critical failure. The baseline draws a smooth, safe curve through the crash, while our GAN accurately tracks the physical cliff.

![Macro vs Micro Anomaly Plot](path/to/your/micro_zoom_graph.png)
*(Replace with your graph showing the baseline smoothing vs the GAN's volatile prediction)*

### 3. Preserving Physical Variance
Under extreme 80% sensor failure, the baseline model's Standard Deviation Ratio collapses. Our GAN successfully resists this collapse, maintaining a highly realistic physical traffic distribution.

![Variance and STD Ratio Comparison](path/to/your/std_ratio_graph.png)
*(Replace with your bar chart or line graph comparing the STD Ratio between the Baseline and GAN across the missing data thresholds)*

---

## Future Scope (Phase 2)
While our Phase 1 Conditional GAN successfully defeats the Mean-Prediction Trap, LSTMs inherently suffer from memory starvation when faced with massive sequential gaps. Phase 2 of this project will replace the recurrent Generator with a **Time-Series Transformer** and **Graph Neural Networks (GNN)**. By utilizing Multi-Head Self-Attention, the model will evaluate the entire time window simultaneously, bypassing the sequential bottleneck to achieve state-of-the-art forecasting at 80% degradation.
