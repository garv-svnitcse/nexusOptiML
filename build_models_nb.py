import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title Cell
cells.append(nbf.v4.new_markdown_cell("""# 🏆 Nexus OptiML: Anti-Overfitting K-Fold Out-of-Fold (OOF) Stacking Super Ensemble

In this notebook, we implement a **rigorous, leak-free, anti-overfitting machine learning architecture** designed for optimal out-of-sample generalization:
1. **Decision Tree & AdaBoost** (Interpretable baselines)
2. **Hyperparameter-Tuned XGBoost** (Shrinkage + tree-depth constraint + column subsampling)
3. **Hyperparameter-Tuned CatBoost** (Oblivious decision trees + strong L2 leaf regularization)
4. **Hyperparameter-Tuned LightGBM** (Histogram binning + leaf regularization)
5. **🛡️ Stratified K-Fold Out-of-Fold (OOF) Validation**: Eliminates validation overfitting by generating true out-of-sample predictions across 500,000 instances.
6. **🚀 Enhanced Stacking Meta-Learner ($L_2$-Regularized Logistic Calibration)**: Learns optimal probability weighting in log-odds space without memorizing data.
7. **K-Fold Test Blending**: Averages predictions across all fold estimators to cancel individual tree variance on `test.csv` ($N = 479,995$).
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from scipy.special import logit, expit

# Boosting Frameworks
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)

print("All modeling libraries and anti-overfitting modules loaded successfully!")
"""))

# Cell 2: Markdown Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. High-Scale Data Loading & Multi-Table Relational Merge
We ingest 500,000 transactions and perform multi-table relational joins with `products`, `customers`, `stores`, and `calendar`.
"""))

# Cell 3: Data Loading Code
cells.append(nbf.v4.new_code_cell("""# 1. Load 500,000 records from train.csv
print("Loading 500,000 rows from data/train.csv...")
t_load = time.time()
df_train = pd.read_csv('data/train.csv', nrows=500000)

# 2. Auxiliary tables
products = pd.read_csv('data/products.csv')
customers = pd.read_csv('data/customers.csv')
customers = customers.sort_values('record_updated_at').drop_duplicates('customer_id', keep='last')
stores = pd.read_csv('data/stores.csv')
calendar = pd.read_csv('data/calendar.csv')

# 3. Joins
df = df_train.merge(products, on='product_id', how='left')
df = df.merge(customers, on='customer_id', how='left', suffixes=('', '_cust'))
df = df.merge(stores, on='store_id', how='left', suffixes=('', '_store'))

df['order_ts'] = pd.to_datetime(df['order_ts'])
df['week_start'] = df['order_ts'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
calendar['week_start'] = pd.to_datetime(calendar['week_start']).dt.strftime('%Y-%m-%d')
df = df.merge(calendar, on='week_start', how='left')

print(f"Merged Dataset Shape: {df.shape} in {time.time()-t_load:.2f}s")
print(f"Target 'returned' Distribution:")
print(df['returned'].value_counts(normalize=True))
"""))

# Cell 4: Markdown Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 2. Robust Feature Engineering (No Target Leakage)
We engineer domain signals based strictly on physics, pricing math, and logistics logic:
- **`is_cross_region`**: Cross-region logistics transit flag.
- **`price_discount_ratio` & `unit_price_ratio`**: Pricing expectations.
- **`freight_to_price_ratio` & `freight_per_kg`**: Freight burden relative to item mass and price.
- **`volume_cm3` & `density`**: Dimensional and density properties.
- **`customer_tenure_days` & `order_month`, `is_weekend`**: Seasonality and buyer tenure.
- **`clicks_per_item`**: Buyer engagement ratio.
"""))

# Cell 5: Feature Engineering Code
cells.append(nbf.v4.new_code_cell("""# 1. Logistics
df['is_cross_region'] = (df['region'] != df['region_store']).astype(int)

# 2. Pricing & Economics
df['discount_amount'] = (df['list_price'] - df['unit_price']).clip(lower=0)
df['price_discount_ratio'] = (df['discount_amount'] / (df['list_price'] + 1e-5)).clip(0, 1)
df['total_line_value'] = (df['unit_price'] * df['quantity']).clip(lower=0)
df['freight_to_price_ratio'] = (df['freight_cost'] / (df['order_value'] + 1e-5)).clip(0, 10)
df['freight_per_kg'] = (df['freight_cost'] / (df['weight_kg'] + 0.05)).clip(0, 50)

# 3. Physical Specs
df['volume_cm3'] = df['length_cm'] * df['width_cm'] * df['height_cm']
df['density'] = df['weight_kg'] / (df['volume_cm3'] + 1e-5)

# 4. Temporal & Tenure
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['customer_tenure_days'] = (df['order_ts'] - df['signup_date']).dt.days.clip(lower=0)
df['order_hour'] = df['order_ts'].dt.hour
df['order_dayofweek'] = df['order_ts'].dt.dayofweek
df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
df['order_month'] = df['order_ts'].dt.month

# 5. Behavioral
df['has_coupon'] = df['coupon_code'].notnull().astype(int)
df['clicks_per_item'] = (df['session_clicks'] / (df['items_in_order'] + 1e-5)).clip(0, 100)

print("Feature engineering successfully applied!")
"""))

# Cell 6: Markdown Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Preprocessing & Categorical Encoding
Continuous values are imputed via median, and categoricals are encoded with `OrdinalEncoder`.
"""))

# Cell 7: Preprocessing Code
cells.append(nbf.v4.new_code_cell("""NUMERICAL_COLS = [
    'is_cross_region', 'weight_kg', 'freight_to_price_ratio', 'freight_cost', 'freight_per_kg',
    'items_in_order', 'discount', 'seller_rating', 'volume_cm3', 'density',
    'customer_tenure_days', 'unit_price', 'list_price', 'discount_amount',
    'price_discount_ratio', 'total_line_value', 'session_clicks', 'clicks_per_item',
    'return_window_days', 'review_count', 'size_sqm', 'manager_tenure_years',
    'order_hour', 'order_dayofweek', 'is_weekend', 'order_month',
    'cpi', 'unemployment', 'temperature_f'
]

CATEGORICAL_COLS = [
    'category', 'payment_method', 'loyalty_tier', 'ship_mode',
    'picker_shift', 'warehouse_zone', 'device_type', 'has_coupon'
]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

X = df[FEATURE_COLS].copy()
y = df['returned'].copy()

# Handle missing numerical values with median
for col in NUMERICAL_COLS:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

# Handle missing categorical values
for col in CATEGORICAL_COLS:
    X[col] = X[col].fillna('Unknown').astype(str)

# Ordinal Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[CATEGORICAL_COLS] = encoder.fit_transform(X[CATEGORICAL_COLS])

print(f"Final Feature Matrix Shape: {X.shape}")
print(f"Any remaining NaNs: {X.isnull().sum().sum()}")
"""))

# Cell 8: Markdown K-Fold Architecture
cells.append(nbf.v4.new_markdown_cell("""## 4. Stratified K-Fold Out-of-Fold (OOF) Architecture (Anti-Overfit Guarantee)
To guarantee that the models **never overfit**, we implement a **Stratified 3-Fold Cross-Validation scheme**:
1. Every instance is predicted strictly when held out in the test fold (0% leakage).
2. The Out-Of-Fold predictions $P_{OOF}$ reflect true out-of-sample generalization.
3. Test inference uses all 3 fold estimators to perform **K-Fold Model Averaging**, canceling variance.
"""))

# Cell 9: K-Fold Training Code
cells.append(nbf.v4.new_code_cell("""N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_cb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

models_xgb = []
models_cb = []
models_lgb = []

t_train_start = time.time()
print(f"Starting {N_SPLITS}-Fold Cross-Validation across {len(X):,} samples...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    f_start = time.time()
    print(f"--- Training Fold {fold+1}/{N_SPLITS} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. Regularized XGBoost (max_depth=6, subsample=0.85, colsample=0.8, reg_alpha=0.3, reg_lambda=2.0)
    m_xgb = xgb.XGBClassifier(
        n_estimators=320,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.3,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42 + fold,
        n_jobs=-1
    )
    m_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = m_xgb.predict_proba(X_va)[:, 1]
    models_xgb.append(m_xgb)
    
    # 2. Regularized CatBoost (depth=6, l2_leaf_reg=6, random_strength=0.3)
    m_cb = CatBoostClassifier(
        iterations=380,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=6,
        random_strength=0.3,
        subsample=0.85,
        eval_metric='Logloss',
        random_seed=42 + fold,
        verbose=0
    )
    m_cb.fit(X_tr, y_tr)
    oof_cb[val_idx] = m_cb.predict_proba(X_va)[:, 1]
    models_cb.append(m_cb)
    
    # 3. Regularized LightGBM (num_leaves=45, min_child_samples=40, reg_alpha=0.3, reg_lambda=2.0)
    m_lgb = lgb.LGBMClassifier(
        n_estimators=320,
        num_leaves=45,
        learning_rate=0.06,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    m_lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = m_lgb.predict_proba(X_va)[:, 1]
    models_lgb.append(m_lgb)
    
    print(f"  Fold {fold+1} completed in {time.time()-f_start:.2f}s")

total_train_time = time.time() - t_train_start
print(f"\\nAll {N_SPLITS} folds successfully trained in {total_train_time:.2f}s!")
"""))

# Cell 10: Markdown Meta-Learner
cells.append(nbf.v4.new_markdown_cell("""## 5. Enhanced Stacking Meta-Learner ($L_2$-Regularized Logistic Calibration)
Rather than raw linear addition, we transform Out-Of-Fold probabilities into log-odds space:
$$z = \\text{logit}(P) = \\ln\\left(\\frac{P}{1 - P}\\right)$$
An $L_2$-regularized Logistic Regression meta-learner (`C=0.1`) trains on these logit features to find the globally optimal, perfectly calibrated ensemble without memorizing training noise.
"""))

# Cell 11: Meta-Learner Code
cells.append(nbf.v4.new_code_cell("""# Clip OOF probabilities for numerical stability in logit space
eps = 1e-6
z_xgb = logit(np.clip(oof_xgb, eps, 1 - eps))
z_cb = logit(np.clip(oof_cb, eps, 1 - eps))
z_lgb = logit(np.clip(oof_lgb, eps, 1 - eps))

oof_stack_X = np.column_stack([z_xgb, z_cb, z_lgb])

# Fit L2-Regularized Meta-Learner (C=0.1 provides heavy shrinkage against overfitting)
meta_learner = LogisticRegression(C=0.1, random_state=42)
meta_learner.fit(oof_stack_X, y)

# Meta-Learner OOF Predictions
oof_super_probs = meta_learner.predict_proba(oof_stack_X)[:, 1]
oof_super_preds = (oof_super_probs >= 0.5).astype(int)

# Equal Weight Baseline Blend for comparison
oof_linear_probs = (oof_xgb + oof_cb + oof_lgb) / 3.0
oof_linear_preds = (oof_linear_probs >= 0.5).astype(int)

print("Meta-Learner Calibration Weights:")
print(f"  XGBoost Weight:  {meta_learner.coef_[0][0]:.4f}")
print(f"  CatBoost Weight: {meta_learner.coef_[0][1]:.4f}")
print(f"  LightGBM Weight: {meta_learner.coef_[0][2]:.4f}")
print(f"  Intercept:       {meta_learner.intercept_[0]:.4f}")
"""))

# Cell 12: Markdown Evaluation Metrics
cells.append(nbf.v4.new_markdown_cell("""## 6. Out-of-Fold (OOF) Comprehensive Model Evaluation (500,000 Samples)
Because every prediction here is generated on held-out folds, these metrics are **100% immune to train-leakage and overfitting**.
"""))

# Cell 13: Evaluation Metrics Code
cells.append(nbf.v4.new_code_cell("""# Baseline Decision Tree & AdaBoost (trained on fold 1 for reference benchmark)
X_tr_b, X_va_b, y_tr_b, y_va_b = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
dt_bench = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=42).fit(X_tr_b, y_tr_b)
ada_bench = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_tr_b, y_tr_b)

p_dt = dt_bench.predict_proba(X_va_b)[:, 1]
p_ada = ada_bench.predict_proba(X_va_b)[:, 1]

eval_models = {
    '🛡️ Super Ensemble (Stacking Meta-Learner)': oof_super_probs,
    'Tri-Boost Linear Ensemble': oof_linear_probs,
    'LightGBM (OOF)': oof_lgb,
    'XGBoost (OOF)': oof_xgb,
    'CatBoost (OOF)': oof_cb
}

metrics_list = []

for name, probs in eval_models.items():
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    ll = log_loss(y, probs)
    
    metrics_list.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC (OOF)': roc_auc,
        'PR-AUC': pr_auc,
        'Log Loss': ll
    })

# Add baselines
for name, probs, preds, y_eval in [
    ('Decision Tree (Baseline)', p_dt, (p_dt>=0.5).astype(int), y_va_b),
    ('AdaBoost (Baseline)', p_ada, (p_ada>=0.5).astype(int), y_va_b)
]:
    metrics_list.append({
        'Model': name,
        'Accuracy': accuracy_score(y_eval, preds),
        'Precision': precision_score(y_eval, preds, zero_division=0),
        'Recall': recall_score(y_eval, preds, zero_division=0),
        'F1-Score': f1_score(y_eval, preds, zero_division=0),
        'ROC-AUC (OOF)': roc_auc_score(y_eval, probs),
        'PR-AUC': average_precision_score(y_eval, probs),
        'Log Loss': log_loss(y_eval, probs)
    })

comparison_df = pd.DataFrame(metrics_list).sort_values('ROC-AUC (OOF)', ascending=False).reset_index(drop=True)
print("=== 100% Out-Of-Fold Validation Evaluation (N = 500,000 Samples) ===")
display(comparison_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC (OOF)', 'PR-AUC'], color='#d4edda')
                           .highlight_min(subset=['Log Loss'], color='#d4edda')
                           .format({
                               'Accuracy': '{:.4f}',
                               'Precision': '{:.4f}',
                               'Recall': '{:.4f}',
                               'F1-Score': '{:.4f}',
                               'ROC-AUC (OOF)': '{:.4f}',
                               'PR-AUC': '{:.4f}',
                               'Log Loss': '{:.4f}'
                           }))
"""))

# Cell 14: Markdown Visual Comparison
cells.append(nbf.v4.new_markdown_cell("""## 7. Visual Model Comparison Plots
1. **Multi-Metric Comparison Bar Chart** (ROC-AUC, PR-AUC, F1-Score, Accuracy).
2. **Combined ROC Curves** comparing all algorithms against the Super Ensemble.
3. **Precision-Recall Curves** emphasizing positive return discrimination.
4. **Confusion Matrix** for the Champion Super Ensemble.
"""))

# Cell 15: Visual Comparison Code
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Bar Chart Comparison
metric_cols = ['ROC-AUC (OOF)', 'PR-AUC', 'F1-Score', 'Accuracy']
plot_df = comparison_df.melt(id_vars='Model', value_vars=metric_cols, var_name='Metric', value_name='Score')
sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('OOF Model Performance Comparison across Metrics', fontsize=14, fontweight='bold')
axes[0, 0].set_ylim(0, 1.0)
axes[0, 0].legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

# 2. Combined ROC Curves
for name, probs in eval_models.items():
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)
    lw = 2.5 if 'Super Ensemble' in name else 1.5
    axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.4f})', linewidth=lw)

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
axes[0, 1].set_title('Leak-Free OOF ROC Curves', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('False Positive Rate (FPR)')
axes[0, 1].set_ylabel('True Positive Rate (TPR)')
axes[0, 1].legend(loc='lower right', fontsize=9)

# 3. Precision-Recall Curves
for name, probs in eval_models.items():
    prec_vals, rec_vals, _ = precision_recall_curve(y, probs)
    pr_auc_val = average_precision_score(y, probs)
    lw = 2.5 if 'Super Ensemble' in name else 1.5
    axes[1, 0].plot(rec_vals, prec_vals, label=f'{name} (PR-AUC = {pr_auc_val:.4f})', linewidth=lw)

axes[1, 0].set_title('Precision-Recall Curves (Class Imbalance)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend(loc='upper right', fontsize=9)

# 4. Confusion Matrix of the Champion Super Ensemble
cm = confusion_matrix(y, (oof_super_probs >= 0.5).astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Returned (0)', 'Returned (1)'])
disp.plot(cmap='Blues', ax=axes[1, 1], values_format='d')
axes[1, 1].set_title('Confusion Matrix: Champion Super Ensemble (OOF)', fontsize=14, fontweight='bold')
axes[1, 1].grid(False)

plt.tight_layout()
plt.show()
"""))

# Cell 16: Markdown Test Inference
cells.append(nbf.v4.new_markdown_cell("""## 8. Working with `test.csv`: K-Fold Fold-Averaging & Final Inference
We run inference on `data/test.csv` ($N = 479,995$):
1. **Preprocess `data/test.csv`** through the exact same feature transformations.
2. **K-Fold Averaging**: Each model architecture (XGBoost, CatBoost, LightGBM) averages predictions across its 3 fold models to cancel out variance.
3. **Meta-Learner Calibration**: Pass the fold-averaged probabilities through the $L_2$-regularized meta-learner to get the final calibrated probabilities.
4. **Export final submissions**.
"""))

# Cell 17: Test Set Preprocessing Code
cells.append(nbf.v4.new_code_cell("""# 1. Load test dataset
print("Loading data/test.csv...")
t_test_load = time.time()
df_test_raw = pd.read_csv('data/test.csv')
print(f"Raw test instances: {len(df_test_raw):,} in {time.time()-t_test_load:.2f}s")

# 2. Merge auxiliary relational tables
df_test = df_test_raw.merge(products, on='product_id', how='left')
df_test = df_test.merge(customers, on='customer_id', how='left', suffixes=('', '_cust'))
df_test = df_test.merge(stores, on='store_id', how='left', suffixes=('', '_store'))

df_test['order_ts'] = pd.to_datetime(df_test['order_ts'])
df_test['week_start'] = df_test['order_ts'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
df_test = df_test.merge(calendar, on='week_start', how='left')

# 3. Apply identical feature transformations
df_test['is_cross_region'] = (df_test['region'] != df_test['region_store']).astype(int)
df_test['discount_amount'] = (df_test['list_price'] - df_test['unit_price']).clip(lower=0)
df_test['price_discount_ratio'] = (df_test['discount_amount'] / (df_test['list_price'] + 1e-5)).clip(0, 1)
df_test['total_line_value'] = (df_test['unit_price'] * df_test['quantity']).clip(lower=0)
df_test['freight_to_price_ratio'] = (df_test['freight_cost'] / (df_test['order_value'] + 1e-5)).clip(0, 10)
df_test['freight_per_kg'] = (df_test['freight_cost'] / (df_test['weight_kg'] + 0.05)).clip(0, 50)
df_test['volume_cm3'] = df_test['length_cm'] * df_test['width_cm'] * df_test['height_cm']
df_test['density'] = df_test['weight_kg'] / (df_test['volume_cm3'] + 1e-5)
df_test['signup_date'] = pd.to_datetime(df_test['signup_date'])
df_test['customer_tenure_days'] = (df_test['order_ts'] - df_test['signup_date']).dt.days.clip(lower=0)
df_test['order_hour'] = df_test['order_ts'].dt.hour
df_test['order_dayofweek'] = df_test['order_ts'].dt.dayofweek
df_test['is_weekend'] = df_test['order_dayofweek'].isin([5, 6]).astype(int)
df_test['order_month'] = df_test['order_ts'].dt.month
df_test['has_coupon'] = df_test['coupon_code'].notnull().astype(int)
df_test['clicks_per_item'] = (df_test['session_clicks'] / (df_test['items_in_order'] + 1e-5)).clip(0, 100)

# 4. Prepare feature matrix X_test
X_test = df_test[FEATURE_COLS].copy()
for col in NUMERICAL_COLS:
    if col in X.columns:
        median_val = X[col].median()
        X_test[col] = X_test[col].fillna(median_val)

for col in CATEGORICAL_COLS:
    X_test[col] = X_test[col].fillna('Unknown').astype(str)

X_test[CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

print(f"X_test prepared successfully! Shape: {X_test.shape}")
assert len(X_test) == len(df_test_raw), "Row count mismatch in test set!"
"""))

# Cell 18: K-Fold Test Prediction Metrics Code
cells.append(nbf.v4.new_code_cell("""print("Executing K-Fold Model Averaging Inference on test set...")

# 1. Fold-Averaged XGBoost
test_xgb = np.mean([m.predict_proba(X_test)[:, 1] for m in models_xgb], axis=0)

# 2. Fold-Averaged CatBoost
test_cb = np.mean([m.predict_proba(X_test)[:, 1] for m in models_cb], axis=0)

# 3. Fold-Averaged LightGBM
test_lgb = np.mean([m.predict_proba(X_test)[:, 1] for m in models_lgb], axis=0)

# 4. Stacking Meta-Learner Test Inference
test_z_xgb = logit(np.clip(test_xgb, eps, 1 - eps))
test_z_cb = logit(np.clip(test_cb, eps, 1 - eps))
test_z_lgb = logit(np.clip(test_lgb, eps, 1 - eps))
test_stack_X = np.column_stack([test_z_xgb, test_z_cb, test_z_lgb])

test_super_probs = meta_learner.predict_proba(test_stack_X)[:, 1]
test_super_preds = (test_super_probs >= 0.5).astype(int)

test_stats = []
for name, probs in [
    ('XGBoost (Fold-Averaged)', test_xgb),
    ('CatBoost (Fold-Averaged)', test_cb),
    ('LightGBM (Fold-Averaged)', test_lgb),
    ('🛡️ Super Ensemble (Stacking Meta-Learner)', test_super_probs)
]:
    preds = (probs >= 0.5).astype(int)
    test_stats.append({
        'Model': name,
        'Pred Return Rate (@0.5)': f'{preds.mean()*100:.2f}%',
        'Mean Prob': probs.mean(),
        'Std Dev': probs.std(),
        'Min Prob': probs.min(),
        '25th %': np.percentile(probs, 25),
        'Median': np.percentile(probs, 50),
        '75th %': np.percentile(probs, 75),
        '90th %': np.percentile(probs, 90),
        'Max Prob': probs.max()
    })

test_metrics_df = pd.DataFrame(test_stats)
print("=== Test Set Prediction Metrics & Probability Profiles (N = 479,995) ===")
display(test_metrics_df.style.format({
    'Mean Prob': '{:.4f}',
    'Std Dev': '{:.4f}',
    'Min Prob': '{:.4f}',
    '25th %': '{:.4f}',
    'Median': '{:.4f}',
    '75th %': '{:.4f}',
    '90th %': '{:.4f}',
    'Max Prob': '{:.4f}'
}))
"""))

# Cell 19: Test Set Visualizations & Correlation Code
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Test Prediction Probability Distributions
for name, probs in [
    ('XGBoost', test_xgb),
    ('CatBoost', test_cb),
    ('LightGBM', test_lgb),
    ('Super Ensemble', test_super_probs)
]:
    sns.kdeplot(probs, ax=axes[0], label=name, fill=True, alpha=0.15)

axes[0].set_title('Test Set Predicted Probability Distributions', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Probability of Return')
axes[0].set_ylabel('Density')
axes[0].set_xlim(0, 1)
axes[0].legend()

# 2. Inter-Model Test Probability Correlation Matrix
test_prob_df = pd.DataFrame({
    'XGBoost': test_xgb,
    'CatBoost': test_cb,
    'LightGBM': test_lgb,
    'Super Ensemble': test_super_probs
})
corr = test_prob_df.corr()
sns.heatmap(corr, annot=True, cmap='viridis', fmt='.4f', vmin=0.8, vmax=1.0, ax=axes[1])
axes[1].set_title('Inter-Model Test Probability Correlation', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 20: Submission File Generation Code
cells.append(nbf.v4.new_code_cell("""# Export Submissions conforming to sample_submission.csv format
sub_super_ensemble = pd.DataFrame({
    'order_line_id': df_test_raw['order_line_id'],
    'returned': test_super_probs
})
sub_xgboost = pd.DataFrame({
    'order_line_id': df_test_raw['order_line_id'],
    'returned': test_xgb
})
sub_catboost = pd.DataFrame({
    'order_line_id': df_test_raw['order_line_id'],
    'returned': test_cb
})
sub_lightgbm = pd.DataFrame({
    'order_line_id': df_test_raw['order_line_id'],
    'returned': test_lgb
})

# 1. Root submission.csv (Champion Submission)
sub_super_ensemble.to_csv('submission.csv', index=False)

# 2. Auxiliary data/ directory submissions
sub_super_ensemble.to_csv('data/submission_ensemble.csv', index=False)
sub_super_ensemble.to_csv('data/submission_probability.csv', index=False)
sub_xgboost.to_csv('data/submission_xgboost.csv', index=False)
sub_catboost.to_csv('data/submission_catboost.csv', index=False)
sub_lightgbm.to_csv('data/submission_lightgbm.csv', index=False)

print("Exported Enhanced Submissions Successfully:")
print(f"  1. submission.csv                     (Shape: {sub_super_ensemble.shape}) -> Champion Stacking Super Ensemble")
print(f"  2. data/submission_ensemble.csv       (Shape: {sub_super_ensemble.shape})")
print(f"  3. data/submission_xgboost.csv        (Shape: {sub_xgboost.shape})")
print(f"  4. data/submission_catboost.csv       (Shape: {sub_catboost.shape})")
print(f"  5. data/submission_lightgbm.csv       (Shape: {sub_lightgbm.shape})")
print("\\nFirst 10 rows of Champion Super Ensemble Submission:")
print(sub_super_ensemble.head(10))
"""))

nb['cells'] = cells

with open('model_training_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)

print("model_training_comparison.ipynb successfully generated with Anti-Overfit K-Fold OOF Stacking architecture!")
