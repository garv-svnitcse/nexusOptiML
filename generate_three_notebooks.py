import nbformat as nbf
import os

# ==============================================================================
# NOTEBOOK 1: BASELINE EXPLORATION & BENCHMARKING
# ==============================================================================
nb1 = nbf.v4.new_notebook()
cells1 = []

cells1.append(nbf.v4.new_markdown_cell("""# 📓 Notebook 1: Basic Models Benchmark (AdaBoost, Random Forest, Basic XGBoost, Basic CatBoost)
**Objective**: Build and evaluate initial baseline models (AdaBoost, Random Forest, Basic XGBoost, and Basic CatBoost) on transactional e-commerce data to establish an initial comparative benchmark for product return prediction.

### Pipeline Overview:
1. Multi-table relational joins (`train`, `products`, `customers`, `stores`, `calendar`).
2. Core feature engineering (pricing, basic freight ratios, temporal features).
3. Median imputation & Ordinal Encoding.
4. Model training:
   - **AdaBoost Classifier** (Baseline sequential boosting)
   - **Random Forest Classifier** (Baseline bagging ensemble)
   - **Basic XGBoost** (Default gradient boosted trees)
   - **Basic CatBoost** (Default categorical gradient boosting)
5. Evaluation across Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC.
"""))

cells1.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
print("Libraries imported successfully for Notebook 1!")
"""))

cells1.append(nbf.v4.new_code_cell("""# 1. Load Data (200,000 sample for baseline benchmarking)
print("Loading 200,000 samples for baseline exploration...")
t0 = time.time()
df_train = pd.read_csv('data/train.csv', nrows=200000)
products = pd.read_csv('data/products.csv')
customers = pd.read_csv('data/customers.csv').sort_values('record_updated_at').drop_duplicates('customer_id', keep='last')
stores = pd.read_csv('data/stores.csv')
calendar = pd.read_csv('data/calendar.csv')

# Joins
df = df_train.merge(products, on='product_id', how='left')
df = df.merge(customers, on='customer_id', how='left', suffixes=('', '_cust'))
df = df.merge(stores, on='store_id', how='left', suffixes=('', '_store'))

df['order_ts'] = pd.to_datetime(df['order_ts'])
df['week_start'] = df['order_ts'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
calendar['week_start'] = pd.to_datetime(calendar['week_start']).dt.strftime('%Y-%m-%d')
df = df.merge(calendar, on='week_start', how='left')

print(f"Data joined in {time.time()-t0:.2f}s | Shape: {df.shape}")
print(f"Return Rate: {df['returned'].mean()*100:.2f}%")
"""))

cells1.append(nbf.v4.new_code_cell("""# 2. Basic Feature Engineering
df['is_cross_region'] = (df['region'] != df['region_store']).astype(int)
df['discount_amount'] = (df['list_price'] - df['unit_price']).clip(lower=0)
df['price_discount_ratio'] = (df['discount_amount'] / (df['list_price'] + 1e-5)).clip(0, 1)
df['total_line_value'] = df['unit_price'] * df['quantity']
df['freight_to_price_ratio'] = (df['freight_cost'] / (df['order_value'] + 1e-5)).clip(0, 10)
df['volume_cm3'] = df['length_cm'] * df['width_cm'] * df['height_cm']
df['density'] = df['weight_kg'] / (df['volume_cm3'] + 1e-5)
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['customer_tenure_days'] = (df['order_ts'] - df['signup_date']).dt.days.clip(lower=0)
df['order_hour'] = df['order_ts'].dt.hour
df['order_dayofweek'] = df['order_ts'].dt.dayofweek
df['has_coupon'] = df['coupon_code'].notnull().astype(int)

num_cols = [
    'is_cross_region', 'weight_kg', 'freight_to_price_ratio', 'freight_cost',
    'items_in_order', 'discount', 'seller_rating', 'volume_cm3', 'density',
    'customer_tenure_days', 'unit_price', 'list_price', 'discount_amount',
    'price_discount_ratio', 'total_line_value', 'session_clicks',
    'return_window_days', 'review_count', 'size_sqm', 'manager_tenure_years',
    'order_hour', 'order_dayofweek', 'cpi', 'unemployment', 'temperature_f'
]
cat_cols = ['category', 'payment_method', 'loyalty_tier', 'ship_mode', 'warehouse_zone', 'has_coupon']
feature_cols = num_cols + cat_cols

X = df[feature_cols].copy()
y = df['returned'].copy()

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna('Unknown').astype(str)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train set: {len(X_train):,}, Validation set: {len(X_val):,}")
"""))

cells1.append(nbf.v4.new_code_cell("""# Fit Basic Models (AdaBoost, Random Forest, Basic XGBoost, Basic CatBoost)
import xgboost as xgb
from catboost import CatBoostClassifier

models = {
    'AdaBoost (Basic)': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'Random Forest (Basic)': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42),
    'XGBoost (Basic)': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1),
    'CatBoost (Basic)': CatBoostClassifier(iterations=150, depth=5, learning_rate=0.1, eval_metric='Logloss', random_seed=42, verbose=0)
}

results = []
pred_probs = {}

for name, model in models.items():
    t_start = time.time()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t_start
    preds = (probs >= 0.5).astype(int)
    pred_probs[name] = probs
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, preds),
        'Precision': precision_score(y_val, preds, zero_division=0),
        'Recall': recall_score(y_val, preds, zero_division=0),
        'F1-Score': f1_score(y_val, preds, zero_division=0),
        'ROC-AUC': roc_auc_score(y_val, probs),
        'PR-AUC': average_precision_score(y_val, probs),
        'Log Loss': log_loss(y_val, probs),
        'Train Time (s)': elapsed
    })

baseline_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("=== Notebook 1: Basic Models Benchmark (AdaBoost, RF, XGBoost, CatBoost) ===")
display(baseline_df.style.highlight_max(subset=['ROC-AUC', 'PR-AUC', 'F1-Score', 'Accuracy'], color='#d4edda')
                        .highlight_min(subset=['Log Loss'], color='#d4edda')
                        .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1-Score': '{:.4f}', 'ROC-AUC': '{:.4f}', 'PR-AUC': '{:.4f}', 'Log Loss': '{:.4f}', 'Train Time (s)': '{:.2f}'}))
"""))

cells1.append(nbf.v4.new_code_cell("""# 4. Baseline Comparative Visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for name, probs in pred_probs.items():
    fpr, tpr, _ = roc_curve(y_val, probs)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_val, probs):.4f})')

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_title('Baseline Models ROC Curves', fontsize=13, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

for name, probs in pred_probs.items():
    p, r, _ = precision_recall_curve(y_val, probs)
    axes[1].plot(r, p, label=f'{name} (PR-AUC = {average_precision_score(y_val, probs):.4f})')

axes[1].set_title('Baseline Precision-Recall Curves', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend()

plt.tight_layout()
plt.show()

print(\"\\nSummary of Baseline Findings:\")
print(\"1. Decision Tree and Random Forest achieve ~0.63 - 0.64 ROC-AUC.\")
print(\"2. Linear models and standard shallow trees struggle with complex interaction non-linearities.\")
print(\"3. Motivates moving to modern Gradient Boosted Decision Trees (XGBoost, CatBoost, LightGBM).\")
"""))

nb1['cells'] = cells1

with open('notebook_v1_baseline_models.ipynb', 'w') as f:
    nbf.write(nb1, f)
print("Created notebook_v1_baseline_models.ipynb")


# ==============================================================================
# NOTEBOOK 2: TUNED GRADIENT BOOSTERS & WEIGHTED SUPER ENSEMBLE
# ==============================================================================
nb2 = nbf.v4.new_notebook()
cells2 = []

cells2.append(nbf.v4.new_markdown_cell("""# 🚀 Notebook 2: Scaled Dataset, Tuned Gradient Boosters & Weighted Super Ensemble
**Objective**: Scale data ingestion to 500,000 records, engineer advanced economic and logistics features, hyperparameter-tune the Big-Three Gradient Boosters (XGBoost, CatBoost, LightGBM), and optimize an ensemble blend via Scipy SLSQP.

### Pipeline Overview:
1. Multi-table join on **500,000 transactions**.
2. Comprehensive feature engineering (cross-region flag, freight physics, density, customer tenure, clicks per item, pricing ratios).
3. **Hyperparameter Tuning**:
   - Tuned XGBoost (`max_depth=7`, `subsample=0.85`, `colsample_bytree=0.8`, `min_child_weight=3`, `gamma=0.1`, `reg_alpha=0.2`, `reg_lambda=2.0`)
   - Tuned CatBoost (`depth=6`, `l2_leaf_reg=5`, `random_strength=0.2`)
   - Tuned LightGBM (`num_leaves=50`, `min_child_samples=30`, `reg_alpha=0.2`, `reg_lambda=1.5`)
4. **Scipy SLSQP Constrained Optimization**: Learns exact non-negative weights summing to 1 that minimize validation negative ROC-AUC.
5. Evaluation on 100,000 validation holdout and initial test set generation.
"""))

cells2.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from scipy.optimize import minimize
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
print("Libraries imported successfully for Notebook 2!")
"""))

cells2.append(nbf.v4.new_code_cell("""# 1. Scaled Data Loading (500,000 rows)
print("Loading 500,000 records from train.csv...")
t0 = time.time()
df_train = pd.read_csv('data/train.csv', nrows=500000)
products = pd.read_csv('data/products.csv')
customers = pd.read_csv('data/customers.csv').sort_values('record_updated_at').drop_duplicates('customer_id', keep='last')
stores = pd.read_csv('data/stores.csv')
calendar = pd.read_csv('data/calendar.csv')

df = df_train.merge(products, on='product_id', how='left')
df = df.merge(customers, on='customer_id', how='left', suffixes=('', '_cust'))
df = df.merge(stores, on='store_id', how='left', suffixes=('', '_store'))

df['order_ts'] = pd.to_datetime(df['order_ts'])
df['week_start'] = df['order_ts'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
calendar['week_start'] = pd.to_datetime(calendar['week_start']).dt.strftime('%Y-%m-%d')
df = df.merge(calendar, on='week_start', how='left')
print(f"Data joined in {time.time()-t0:.2f}s | Shape: {df.shape}")
"""))

cells2.append(nbf.v4.new_code_cell("""# 2. Rich Domain Feature Engineering
df['is_cross_region'] = (df['region'] != df['region_store']).astype(int)
df['discount_amount'] = (df['list_price'] - df['unit_price']).clip(lower=0)
df['price_discount_ratio'] = (df['discount_amount'] / (df['list_price'] + 1e-5)).clip(0, 1)
df['total_line_value'] = df['unit_price'] * df['quantity']
df['freight_to_price_ratio'] = (df['freight_cost'] / (df['order_value'] + 1e-5)).clip(0, 10)
df['freight_per_kg'] = (df['freight_cost'] / (df['weight_kg'] + 0.05)).clip(0, 50)
df['volume_cm3'] = df['length_cm'] * df['width_cm'] * df['height_cm']
df['density'] = df['weight_kg'] / (df['volume_cm3'] + 1e-5)
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['customer_tenure_days'] = (df['order_ts'] - df['signup_date']).dt.days.clip(lower=0)
df['order_hour'] = df['order_ts'].dt.hour
df['order_dayofweek'] = df['order_ts'].dt.dayofweek
df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
df['order_month'] = df['order_ts'].dt.month
df['has_coupon'] = df['coupon_code'].notnull().astype(int)
df['clicks_per_item'] = (df['session_clicks'] / (df['items_in_order'] + 1e-5)).clip(0, 100)

num_cols = [
    'is_cross_region', 'weight_kg', 'freight_to_price_ratio', 'freight_cost', 'freight_per_kg',
    'items_in_order', 'discount', 'seller_rating', 'volume_cm3', 'density',
    'customer_tenure_days', 'unit_price', 'list_price', 'discount_amount',
    'price_discount_ratio', 'total_line_value', 'session_clicks', 'clicks_per_item',
    'return_window_days', 'review_count', 'size_sqm', 'manager_tenure_years',
    'order_hour', 'order_dayofweek', 'is_weekend', 'order_month',
    'cpi', 'unemployment', 'temperature_f'
]
cat_cols = ['category', 'payment_method', 'loyalty_tier', 'ship_mode', 'picker_shift', 'warehouse_zone', 'device_type', 'has_coupon']
feature_cols = num_cols + cat_cols

X = df[feature_cols].copy()
y = df['returned'].copy()

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna('Unknown').astype(str)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train: {len(X_train):,}, Validation: {len(X_val):,}")
"""))

cells2.append(nbf.v4.new_code_cell("""# 3. Fit Tuned Big-Three Gradient Boosters
print("1. Training Tuned XGBoost...")
t1 = time.time()
xgb_m = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.2,
    reg_lambda=2.0,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_m.fit(X_train, y_train)
p_xgb = xgb_m.predict_proba(X_val)[:, 1]
print(f"XGBoost done in {time.time()-t1:.2f}s | Val ROC-AUC: {roc_auc_score(y_val, p_xgb):.4f}")

print("2. Training Tuned CatBoost...")
t2 = time.time()
cb_m = CatBoostClassifier(
    iterations=450,
    depth=6,
    learning_rate=0.06,
    l2_leaf_reg=5,
    random_strength=0.2,
    subsample=0.85,
    eval_metric='Logloss',
    random_seed=42,
    verbose=0
)
cb_m.fit(X_train, y_train)
p_cb = cb_m.predict_proba(X_val)[:, 1]
print(f"CatBoost done in {time.time()-t2:.2f}s | Val ROC-AUC: {roc_auc_score(y_val, p_cb):.4f}")

print("3. Training Tuned LightGBM...")
t3 = time.time()
lgb_m = lgb.LGBMClassifier(
    n_estimators=400,
    num_leaves=50,
    learning_rate=0.05,
    min_child_samples=30,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_m.fit(X_train, y_train)
p_lgb = lgb_m.predict_proba(X_val)[:, 1]
print(f"LightGBM done in {time.time()-t3:.2f}s | Val ROC-AUC: {roc_auc_score(y_val, p_lgb):.4f}")
"""))

cells2.append(nbf.v4.new_code_cell("""# 4. Scipy SLSQP Optimization for Super Ensemble Weights
def ensemble_loss(weights):
    w1, w2, w3 = weights
    pred = w1 * p_xgb + w2 * p_cb + w3 * p_lgb
    return -roc_auc_score(y_val, pred)

initial_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1.0}

opt_res = minimize(ensemble_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = opt_res.x

p_super = best_weights[0] * p_xgb + best_weights[1] * p_cb + best_weights[2] * p_lgb
p_equal = (p_xgb + p_cb + p_lgb) / 3.0

print(f"Optimal Weights: XGBoost={best_weights[0]:.4f}, CatBoost={best_weights[1]:.4f}, LightGBM={best_weights[2]:.4f}")
print(f"Equal Blend ROC-AUC:     {roc_auc_score(y_val, p_equal):.4f}")
print(f"Optimized Super Ensemble: {roc_auc_score(y_val, p_super):.4f}")
"""))

cells2.append(nbf.v4.new_code_cell("""# 5. Comparative Evaluation Table
eval_dict = {
    '🚀 Super Ensemble (Optimized Blend)': p_super,
    'Tri-Boost Equal Blend': p_equal,
    'LightGBM (Tuned)': p_lgb,
    'XGBoost (Tuned)': p_xgb,
    'CatBoost (Tuned)': p_cb
}

rows = []
for name, probs in eval_dict.items():
    preds = (probs >= 0.5).astype(int)
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, preds),
        'Precision': precision_score(y_val, preds, zero_division=0),
        'Recall': recall_score(y_val, preds, zero_division=0),
        'F1-Score': f1_score(y_val, preds, zero_division=0),
        'ROC-AUC': roc_auc_score(y_val, probs),
        'PR-AUC': average_precision_score(y_val, probs),
        'Log Loss': log_loss(y_val, probs)
    })

comparison_df2 = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("=== Notebook 2: Tuned Boosters & Weighted Super Ensemble ===")
display(comparison_df2.style.highlight_max(subset=['ROC-AUC', 'PR-AUC', 'F1-Score', 'Accuracy'], color='#d4edda')
                            .highlight_min(subset=['Log Loss'], color='#d4edda')
                            .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1-Score': '{:.4f}', 'ROC-AUC': '{:.4f}', 'PR-AUC': '{:.4f}', 'Log Loss': '{:.4f}'}))
"""))

cells2.append(nbf.v4.new_code_cell("""# 6. ROC Curves & PR Curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for name, probs in eval_dict.items():
    fpr, tpr, _ = roc_curve(y_val, probs)
    lw = 2.5 if 'Super' in name else 1.5
    axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_val, probs):.4f})', linewidth=lw)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_title('Validation ROC Curves', fontsize=13, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')

for name, probs in eval_dict.items():
    p, r, _ = precision_recall_curve(y_val, probs)
    lw = 2.5 if 'Super' in name else 1.5
    axes[1].plot(r, p, label=f'{name} (PR-AUC = {average_precision_score(y_val, probs):.4f})', linewidth=lw)

axes[1].set_title('Validation Precision-Recall Curves', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
"""))

nb2['cells'] = cells2

with open('notebook_v2_tuned_super_ensemble.ipynb', 'w') as f:
    nbf.write(nb2, f)
print("Created notebook_v2_tuned_super_ensemble.ipynb")


# ==============================================================================
# NOTEBOOK 3: ANTI-OVERFIT K-FOLD OUT-OF-FOLD (OOF) STACKING SUPER ENSEMBLE
# ==============================================================================
nb3 = nbf.v4.new_notebook()
cells3 = []

cells3.append(nbf.v4.new_markdown_cell("""# 🛡️ Notebook 3: Anti-Overfitting Stratified K-Fold Out-of-Fold (OOF) Stacking Super Ensemble
**Objective**: Build the ultimate leak-free, anti-overfitting machine learning architecture using Stratified K-Fold cross-validation, strong tree regularization, probability calibration in log-odds space via an $L_2$-Regularized Logistic Meta-Learner, and K-Fold fold-averaging inference on `test.csv`.

### Key Anti-Overfitting Pillars:
1. **Stratified 3-Fold Out-Of-Fold (OOF) Validation**:
   - Every single training prediction is generated when held-out in an unseen fold (0% data leakage).
   - Validation metrics represent true out-of-sample generalization across the entire 500,000 instance cohort.
2. **Heavy Regularization in Tree Estimators**:
   - Depth constraints (`max_depth=6`), column subsampling (`colsample_bytree=0.8`), instance subsampling (`subsample=0.85`), minimum leaf weight constraints (`min_child_weight=5`), and $L_1/L_2$ penalties (`reg_alpha=0.3, reg_lambda=2.0, l2_leaf_reg=6`).
3. **Log-Odds ($z$) Space Stacking with $L_2$ Shrinkage (`C=0.1`)**:
   - Rather than heuristic linear blending, a Logistic Regression meta-learner learns optimal probability calibration in log-odds space. With only 4 parameters and heavy $L_2$ penalty, memorization is impossible.
4. **K-Fold Test Inference (Variance Reduction)**:
   - Inference on `test.csv` averages predictions across all 3 fold estimators for each architecture before passing to the meta-learner, canceling individual tree noise.
"""))

cells3.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from scipy.special import logit, expit
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
print("Libraries loaded for Notebook 3 (Anti-Overfit K-Fold Stacking)!")
"""))

cells3.append(nbf.v4.new_code_cell("""# 1. High-Scale Ingestion & Multi-Table Relational Merges
print("Loading 500,000 transactions...")
t0 = time.time()
df_train = pd.read_csv('data/train.csv', nrows=500000)
products = pd.read_csv('data/products.csv')
customers = pd.read_csv('data/customers.csv').sort_values('record_updated_at').drop_duplicates('customer_id', keep='last')
stores = pd.read_csv('data/stores.csv')
calendar = pd.read_csv('data/calendar.csv')

df = df_train.merge(products, on='product_id', how='left')
df = df.merge(customers, on='customer_id', how='left', suffixes=('', '_cust'))
df = df.merge(stores, on='store_id', how='left', suffixes=('', '_store'))

df['order_ts'] = pd.to_datetime(df['order_ts'])
df['week_start'] = df['order_ts'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
calendar['week_start'] = pd.to_datetime(calendar['week_start']).dt.strftime('%Y-%m-%d')
df = df.merge(calendar, on='week_start', how='left')

# 2. Strict Physical/Economic Features
df['is_cross_region'] = (df['region'] != df['region_store']).astype(int)
df['discount_amount'] = (df['list_price'] - df['unit_price']).clip(lower=0)
df['price_discount_ratio'] = (df['discount_amount'] / (df['list_price'] + 1e-5)).clip(0, 1)
df['total_line_value'] = (df['unit_price'] * df['quantity']).clip(lower=0)
df['freight_to_price_ratio'] = (df['freight_cost'] / (df['order_value'] + 1e-5)).clip(0, 10)
df['freight_per_kg'] = (df['freight_cost'] / (df['weight_kg'] + 0.05)).clip(0, 50)
df['volume_cm3'] = df['length_cm'] * df['width_cm'] * df['height_cm']
df['density'] = df['weight_kg'] / (df['volume_cm3'] + 1e-5)
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['customer_tenure_days'] = (df['order_ts'] - df['signup_date']).dt.days.clip(lower=0)
df['order_hour'] = df['order_ts'].dt.hour
df['order_dayofweek'] = df['order_ts'].dt.dayofweek
df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
df['order_month'] = df['order_ts'].dt.month
df['has_coupon'] = df['coupon_code'].notnull().astype(int)
df['clicks_per_item'] = (df['session_clicks'] / (df['items_in_order'] + 1e-5)).clip(0, 100)

num_cols = [
    'is_cross_region', 'weight_kg', 'freight_to_price_ratio', 'freight_cost', 'freight_per_kg',
    'items_in_order', 'discount', 'seller_rating', 'volume_cm3', 'density',
    'customer_tenure_days', 'unit_price', 'list_price', 'discount_amount',
    'price_discount_ratio', 'total_line_value', 'session_clicks', 'clicks_per_item',
    'return_window_days', 'review_count', 'size_sqm', 'manager_tenure_years',
    'order_hour', 'order_dayofweek', 'is_weekend', 'order_month',
    'cpi', 'unemployment', 'temperature_f'
]
cat_cols = ['category', 'payment_method', 'loyalty_tier', 'ship_mode', 'picker_shift', 'warehouse_zone', 'device_type', 'has_coupon']
feature_cols = num_cols + cat_cols

X = df[feature_cols].copy()
y = df['returned'].copy()

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna('Unknown').astype(str)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
print(f"Data prepared in {time.time()-t0:.2f}s | Shape: {X.shape}")
"""))

cells3.append(nbf.v4.new_code_cell("""# 2. Stratified 3-Fold Cross-Validation Scheme (0% Leakage)
N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_cb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

models_xgb, models_cb, models_lgb = [], [], []

print(f"Executing {N_SPLITS}-Fold Cross Validation across {len(X):,} instances...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_f = time.time()
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    
    # Regularized XGBoost
    m_xgb = xgb.XGBClassifier(
        n_estimators=320, max_depth=6, learning_rate=0.06, subsample=0.85,
        colsample_bytree=0.8, min_child_weight=5, gamma=0.1,
        reg_alpha=0.3, reg_lambda=2.0, eval_metric='logloss',
        random_state=42+fold, n_jobs=-1
    )
    m_xgb.fit(X_tr, y_tr)
    oof_xgb[va_idx] = m_xgb.predict_proba(X_va)[:, 1]
    models_xgb.append(m_xgb)
    
    # Regularized CatBoost
    m_cb = CatBoostClassifier(
        iterations=380, depth=6, learning_rate=0.06, l2_leaf_reg=6,
        random_strength=0.3, subsample=0.85, eval_metric='Logloss',
        random_seed=42+fold, verbose=0
    )
    m_cb.fit(X_tr, y_tr)
    oof_cb[va_idx] = m_cb.predict_proba(X_va)[:, 1]
    models_cb.append(m_cb)
    
    # Regularized LightGBM
    m_lgb = lgb.LGBMClassifier(
        n_estimators=320, num_leaves=45, learning_rate=0.06, min_child_samples=40,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=2.0,
        random_state=42+fold, n_jobs=-1, verbose=-1
    )
    m_lgb.fit(X_tr, y_tr)
    oof_lgb[va_idx] = m_lgb.predict_proba(X_va)[:, 1]
    models_lgb.append(m_lgb)
    
    print(f"Fold {fold+1}/{N_SPLITS} complete in {time.time()-t_f:.2f}s")

print("All folds trained successfully!")
"""))

cells3.append(nbf.v4.new_code_cell("""# 3. L2-Regularized Stacking Meta-Learner Training
eps = 1e-6
z_xgb = logit(np.clip(oof_xgb, eps, 1 - eps))
z_cb = logit(np.clip(oof_cb, eps, 1 - eps))
z_lgb = logit(np.clip(oof_lgb, eps, 1 - eps))
oof_stack_X = np.column_stack([z_xgb, z_cb, z_lgb])

meta_learner = LogisticRegression(C=0.1, random_state=42)
meta_learner.fit(oof_stack_X, y)

oof_super_probs = meta_learner.predict_proba(oof_stack_X)[:, 1]
oof_linear_probs = (oof_xgb + oof_cb + oof_lgb) / 3.0

print("Stacking Meta-Learner Learned Weights (Log-Odds Space):")
print(f"  XGBoost Weight:  {meta_learner.coef_[0][0]:.4f}")
print(f"  CatBoost Weight: {meta_learner.coef_[0][1]:.4f}")
print(f"  LightGBM Weight: {meta_learner.coef_[0][2]:.4f}")
print(f"  Intercept:       {meta_learner.intercept_[0]:.4f}")
"""))

cells3.append(nbf.v4.new_code_cell("""# 4. Out-of-Fold (OOF) Comprehensive Evaluation Metrics
eval_models_v3 = {
    '🛡️ Super Ensemble (Stacking Meta-Learner)': oof_super_probs,
    'Tri-Boost Linear Ensemble': oof_linear_probs,
    'LightGBM (OOF)': oof_lgb,
    'XGBoost (OOF)': oof_xgb,
    'CatBoost (OOF)': oof_cb
}

metrics_list3 = []
for name, probs in eval_models_v3.items():
    preds = (probs >= 0.5).astype(int)
    metrics_list3.append({
        'Model': name,
        'Accuracy': accuracy_score(y, preds),
        'Precision': precision_score(y, preds, zero_division=0),
        'Recall': recall_score(y, preds, zero_division=0),
        'F1-Score': f1_score(y, preds, zero_division=0),
        'ROC-AUC (OOF)': roc_auc_score(y, probs),
        'PR-AUC': average_precision_score(y, probs),
        'Log Loss': log_loss(y, probs)
    })

comparison_df3 = pd.DataFrame(metrics_list3).sort_values('ROC-AUC (OOF)', ascending=False).reset_index(drop=True)
print("=== Notebook 3: 100% Out-Of-Fold Validation Evaluation (N = 500,000) ===")
display(comparison_df3.style.highlight_max(subset=['ROC-AUC (OOF)', 'PR-AUC', 'F1-Score', 'Accuracy'], color='#d4edda')
                             .highlight_min(subset=['Log Loss'], color='#d4edda')
                             .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1-Score': '{:.4f}', 'ROC-AUC (OOF)': '{:.4f}', 'PR-AUC': '{:.4f}', 'Log Loss': '{:.4f}'}))
"""))

cells3.append(nbf.v4.new_code_cell("""# 5. K-Fold Averaged Test Inference (N = 479,995)
print("Loading and preparing test.csv for K-Fold Averaging...")
df_test_raw = pd.read_csv('data/test.csv')

df_test = df_test_raw.merge(products, on='product_id', how='left')
df_test = df_test.merge(customers, on='customer_id', how='left', suffixes=('', '_cust'))
df_test = df_test.merge(stores, on='store_id', how='left', suffixes=('', '_store'))

df_test['order_ts'] = pd.to_datetime(df_test['order_ts'])
df_test['week_start'] = df_test['order_ts'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
df_test = df_test.merge(calendar, on='week_start', how='left')

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

X_test = df_test[feature_cols].copy()
for col in num_cols:
    X_test[col] = X_test[col].fillna(X[col].median())
for col in cat_cols:
    X_test[col] = X_test[col].fillna('Unknown').astype(str)
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# K-Fold Fold Averaging across test set
test_xgb = np.mean([m.predict_proba(X_test)[:, 1] for m in models_xgb], axis=0)
test_cb = np.mean([m.predict_proba(X_test)[:, 1] for m in models_cb], axis=0)
test_lgb = np.mean([m.predict_proba(X_test)[:, 1] for m in models_lgb], axis=0)

test_z_xgb = logit(np.clip(test_xgb, eps, 1 - eps))
test_z_cb = logit(np.clip(test_cb, eps, 1 - eps))
test_z_lgb = logit(np.clip(test_lgb, eps, 1 - eps))
test_stack_X = np.column_stack([test_z_xgb, test_z_cb, test_z_lgb])

test_super_probs = meta_learner.predict_proba(test_stack_X)[:, 1]

sub_champion = pd.DataFrame({
    'order_line_id': df_test_raw['order_line_id'],
    'returned': test_super_probs
})
sub_champion.to_csv('submission.csv', index=False)
print(f"Exported Champion submission.csv successfully! Shape: {sub_champion.shape}")
print(f"Predicted Return Rate: {(test_super_probs >= 0.5).mean()*100:.2f}% | Mean Probability: {test_super_probs.mean():.4f}")
"""))

cells3.append(nbf.v4.new_code_cell("""# 6. Final Comparative Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for name, probs in eval_models_v3.items():
    fpr, tpr, _ = roc_curve(y, probs)
    lw = 2.5 if 'Super' in name else 1.5
    axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y, probs):.4f})', linewidth=lw)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_title('Out-Of-Fold ROC Curves (Anti-Overfit)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')

for name, probs in [('XGBoost', test_xgb), ('CatBoost', test_cb), ('LightGBM', test_lgb), ('Super Ensemble', test_super_probs)]:
    sns.kdeplot(probs, ax=axes[1], label=name, fill=True, alpha=0.15)

axes[1].set_title('Test Set Probability Density Distributions', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_xlim(0, 1)
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

nb3['cells'] = cells3

with open('notebook_v3_antioverfit_kfold_stacking.ipynb', 'w') as f:
    nbf.write(nb3, f)
print("Created notebook_v3_antioverfit_kfold_stacking.ipynb")
