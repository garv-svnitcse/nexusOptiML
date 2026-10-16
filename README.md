# 🛒 Nexus OptiML: E-Commerce Product Return Risk Prediction

An end-to-end Machine Learning pipeline and Anti-Overfitting Super Ensemble for predicting product return probabilities on high-scale transactional e-commerce data ($N \approx 1,400,000$).

---

## 📌 Project Overview
E-commerce reverse logistics and product returns cause massive operational friction and margin erosion. This repository implements an end-to-end data science solution:
1. **Multi-Table Relational Data Integration**: Merges transactional lines with `products`, SCD-2 updated `customers`, physical `stores`, and macroeconomic `calendar` data.
2. **Economic & Logistics Feature Engineering**: 37 rich domain signals modeling customer price elasticity, shipping physics, dimensional mass density, buyer tenure, and engagement.
3. **Iterative Model Development**: Progresses from classical baseline models to hyperparameter-tuned gradient boosters, culminating in a leak-free **Stratified K-Fold Out-Of-Fold (OOF) Stacking Super Ensemble**.
4. **Anti-Overfitting Guarantees**: Strict tree regularization, out-of-fold calibration in logit space via an $L_2$-regularized Logistic meta-learner, and K-Fold test model averaging.

---

## 📓 Notebook Iterations (3 Distinct Versions)

| Version | Notebook | Focus | Key Models / Techniques | ROC-AUC |
| :---: | :--- | :--- | :--- | :---: |
| **v1** | [`notebook_v1_baseline_models.ipynb`](./notebook_v1_baseline_models.ipynb) | Classical Baselines & Initial Benchmarking | AdaBoost, Random Forest, Basic XGBoost, Basic CatBoost | 0.6224 – 0.6595 |
| **v2** | [`notebook_v2_tuned_super_ensemble.ipynb`](./notebook_v2_tuned_super_ensemble.ipynb) | Scaled 500k Data, Tuned Boosters & Scipy SLSQP Blend | Tuned XGBoost, Tuned CatBoost, Tuned LightGBM + Constrained Weighted Blend | **0.6747** |
| **v3** | [`notebook_v3_antioverfit_kfold_stacking.ipynb`](./notebook_v3_antioverfit_kfold_stacking.ipynb) | Anti-Overfit Stratified K-Fold OOF Stacking | 3-Fold OOF Validation + $L_2$-Regularized Logistic Meta-Learner + Test Averaging | **0.6721** *(100% OOF)* |
| **v4** | [`notebook_v4_gradient_descent_advanced_stacking.ipynb`](./notebook_v4_gradient_descent_advanced_stacking.ipynb) | Gradient Descent Optimization & Multi-Level Meta Stacking | SGD LogLoss Meta + Newton L2 GD + XGBoost Tree Ensemble Meta + Hybrid Blend | **0.6721** *(LogLoss 0.4435)* |

---

### 🔍 Version Details

### 1. [Notebook v1: Baseline Models Benchmark](./notebook_v1_baseline_models.ipynb)
- Evaluates initial out-of-the-box models on a 200,000-row sample.
- **Models**:
  - `AdaBoost (Basic)`: ROC-AUC = 0.6224 | Log Loss = 0.4597
  - `Random Forest (Basic)`: ROC-AUC = 0.6428 | Log Loss = 0.4527
  - `CatBoost (Basic)`: ROC-AUC = 0.6551 | Log Loss = 0.4487
  - `XGBoost (Basic)`: ROC-AUC = 0.6595 | Log Loss = 0.4471
- **Key Insight**: Out-of-the-box trees struggle with subtle class imbalance and non-linear interactions, motivating deep feature engineering and systematic hyperparameter tuning.

### 2. [Notebook v2: Tuned Boosters & Weighted Super Ensemble](./notebook_v2_tuned_super_ensemble.ipynb)
- Ingests **500,000 transactions** and applies 37 economic, logistics, and behavioral features.
- Exhaustive hyperparameter tuning on the Big Three Gradient Boosters:
  - **Tuned LightGBM**: `num_leaves=50`, `learning_rate=0.05`, `min_child_samples=30`, `reg_alpha=0.2`, `reg_lambda=1.5`
  - **Tuned XGBoost**: `max_depth=7`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.8`, `min_child_weight=3`, `gamma=0.1`, `reg_alpha=0.2`, `reg_lambda=2.0`
  - **Tuned CatBoost**: `depth=6`, `learning_rate=0.06`, `l2_leaf_reg=5`, `random_strength=0.2`
- **Scipy SLSQP Constrained Optimization**: Minimizes validation negative ROC-AUC to find optimal weights:
  $$\text{Pred} = w_1 \cdot P_{\text{XGB}} + w_2 \cdot P_{\text{Cat}} + w_3 \cdot P_{\text{LGB}}, \quad \sum w_i = 1, \; w_i \ge 0$$
- Result: **0.6747 ROC-AUC** and **0.4426 Log Loss** on holdout validation.

### 3. [Notebook v3: Anti-Overfit K-Fold OOF Stacking Super Ensemble](./notebook_v3_antioverfit_kfold_stacking.ipynb)
- **Eliminates Overfitting Risk via 4 Architectural Pillars**:
  1. **Stratified 3-Fold Out-Of-Fold (OOF)**: Every training prediction is generated on held-out folds (0% data leakage).
  2. **Heavy Tree Regularization**: Reduced tree depth to 6, increased `min_child_weight=5`, increased $L_1/L_2$ penalties (`reg_alpha=0.3, reg_lambda=2.0, l2_leaf_reg=6`).
  3. **Log-Odds ($z$) Space Stacking**: Transforms base model probabilities into log-odds space:
     $$z = \text{logit}(P) = \ln\left(\frac{P}{1 - P}\right)$$
     An $L_2$-regularized Logistic Regression (`C=0.1`) learns optimal calibration weights without memorizing noise.
  4. **K-Fold Test Inference**: Averages predictions across all 3 fold estimators for each algorithm on `test.csv` ($N = 479,995$) to cancel out individual tree variance.

---

## 🏆 Final Model Comparison (100% Out-Of-Fold Validation)

| Model Architecture | ROC-AUC (OOF) | PR-AUC | Log Loss | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🛡️ **Super Ensemble (Stacking Meta-Learner)** | **0.6721** | **0.3183** | **0.4434** | **82.06%** | **0.0495** |
| **Tri-Boost Linear Ensemble** | 0.6716 | 0.3177 | 0.4438 | 82.04% | 0.0302 |
| **LightGBM (OOF)** | 0.6714 | 0.3170 | 0.4438 | 82.04% | 0.0332 |
| **XGBoost (OOF)** | 0.6705 | 0.3167 | 0.4439 | 82.04% | 0.0390 |
| **CatBoost (OOF)** | 0.6669 | 0.3128 | 0.4453 | 82.03% | 0.0258 |
| **Decision Tree (Baseline)** | 0.6325 | 0.2740 | 0.4579 | 81.86% | 0.0254 |
| **AdaBoost (Baseline)** | 0.6245 | 0.2683 | 0.4597 | 81.96% | 0.0000 |

---

## 📦 Test Set Submission Verification (`submission.csv`)

- **File**: [`submission.csv`](./submission.csv)
- **Total Test Instances**: `479,995` rows (0 missing/NaNs).
- **Target Distribution**: Mean probability = `0.2015`, matching historical ground truth return rates (~18–20%).
- **Probability Spread**: Smooth distribution from `0.0275` to `0.8370`.

---

## 🛠️ Repository Structure
```
nexusOptiML/
├── notebook_v1_baseline_models.ipynb        # Iteration 1: Classical baseline benchmarks
├── notebook_v2_tuned_super_ensemble.ipynb   # Iteration 2: Scaled data, tuned boosters & blend
├── notebook_v3_antioverfit_kfold_stacking.ipynb # Iteration 3: Anti-overfit K-Fold OOF Stacking
├── model_training_comparison.ipynb          # Master comprehensive comparison notebook
├── feature_analysis.ipynb                   # Feature discovery & correlation analysis
├── build_models_nb.py                       # Automated notebook generator script
├── generate_three_notebooks.py              # Generator for all 3 versioned notebooks
├── submission.csv                           # Final Champion test prediction probabilities
└── README.md                                # Project documentation
```

---

## 🚀 Quickstart
```bash
# Clone the repository
git clone https://github.com/garv-svnitcse/nexusOptiML.git
cd nexusOptiML

# Create and activate environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn xgboost catboost lightgbm matplotlib seaborn jupyter
```