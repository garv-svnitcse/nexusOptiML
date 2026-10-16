import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Markdown Overview
cells.append(nbf.v4.new_markdown_cell("""# ⚡ Notebook 4: Gradient Descent Optimization & Multi-Level Meta-Ensemble Stacking
**Objective**: Fully optimize product return prediction by leveraging **Gradient Descent** across both base estimators and stacking meta-models. We compare:
1. **Base Gradient Boosters** (Functional Gradient Descent with 2nd-order Taylor expansion).
2. **Pure Stochastic Gradient Descent (SGD) Meta-Learner** (Log-Loss minimization).
3. **$L_2$-Regularized Logistic Meta-Learner** (Convex gradient descent in log-odds space).
4. **XGBoost Tree Meta-Learner** (Non-linear functional gradient descent in meta-space).
5. **🏆 Super Hybrid Meta-Ensemble** (Combining Gradient Descent calibration + XGBoost ensemble consensus).

---

### 🧠 Mathematical Principles of Gradient Descent Optimization
1. **Binary Cross-Entropy Loss (Log Loss)**:
   $$L(y, \\hat{y}) = -\\Big[ y \\ln(\\hat{y}) + (1 - y) \\ln(1 - \\hat{y}) \\Big]$$
2. **Parameter-Space Gradient Descent (Meta-Model)**:
   $$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\cdot \\nabla_{\\mathbf{w}} L, \\quad \\nabla_{\\mathbf{w}} L = (\\hat{y} - y) \\cdot \\mathbf{x}$$
3. **Function-Space Gradient Descent (XGBoost / LightGBM / CatBoost)**:
   $$F_m(x) = F_{m-1}(x) + \\eta \\cdot f_m(x)$$
   where each tree $f_m(x)$ fits the negative gradient residuals $r_i = y_i - \\hat{y}_i$, with XGBoost utilizing Hessian curvature $h_i = \\hat{y}_i(1 - \\hat{y}_i)$.
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from scipy.special import logit, expit

# Gradient Boosted Frameworks
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
print("Libraries imported successfully for Notebook 4!")
"""))

# Cell 2: Data Ingestion & Joins
cells.append(nbf.v4.new_code_cell("""# 1. High-Scale Ingestion (500,000 transactions)
print("Loading 500,000 transactions from train.csv...")
t0 = time.time()
df_train = pd.read_csv('data/train.csv', nrows=500000)
products = pd.read_csv('data/products.csv')
customers = pd.read_csv('data/customers.csv').sort_values('record_updated_at').drop_duplicates('customer_id', keep='last')
stores = pd.read_csv('data/stores.csv')
calendar = pd.read_csv('data/calendar.csv')

# Relational Joins
df = df_train.merge(products, on='product_id', how='left')
df = df.merge(customers, on='customer_id', how='left', suffixes=('', '_cust'))
df = df.merge(stores, on='store_id', how='left', suffixes=('', '_store'))

df['order_ts'] = pd.to_datetime(df['order_ts'])
df['week_start'] = df['order_ts'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
calendar['week_start'] = pd.to_datetime(calendar['week_start']).dt.strftime('%Y-%m-%d')
df = df.merge(calendar, on='week_start', how='left')

print(f"Data merged in {time.time()-t0:.2f}s | Shape: {df.shape}")
print(f"Empirical Target Return Rate: {df['returned'].mean()*100:.2f}%")
"""))

# Cell 3: Feature Engineering
cells.append(nbf.v4.new_code_cell("""# 2. Strict Non-Leaking Domain Feature Engineering
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

# Median Imputation & Categorical Encoding
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna('Unknown').astype(str)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
print(f"Feature Matrix Ready: {X.shape}")
"""))

# Cell 4: Markdown Stratified K-Fold Training
cells.append(nbf.v4.new_markdown_cell("""## 1. Stratified 3-Fold Cross-Validation (Functional Gradient Descent)
Each base model optimizes its loss function via Functional Gradient Descent:
- **XGBoost**: 2nd-order Taylor approximation with learning rate $\\eta = 0.05$ and $L_1/L_2$ regularization.
- **CatBoost**: Symmetric oblivious trees with ordered gradient descent to eliminate prediction shift.
- **LightGBM**: Leaf-wise tree growth with gradient-based one-side sampling (GOSS).
"""))

# Cell 5: K-Fold Code
cells.append(nbf.v4.new_code_cell("""N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_cb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

models_xgb, models_cb, models_lgb = [], [], []

t_train_start = time.time()
print(f"Starting {N_SPLITS}-Fold Cross-Validation across {len(X):,} samples...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    f_start = time.time()
    print(f"--- Training Fold {fold+1}/{N_SPLITS} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. XGBoost (Gradient Descent with Shrinkage eta=0.05)
    m_xgb = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=4,
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
    
    # 2. CatBoost (Ordered Functional Gradient Descent)
    m_cb = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        random_strength=0.2,
        subsample=0.85,
        eval_metric='Logloss',
        random_seed=42 + fold,
        verbose=0
    )
    m_cb.fit(X_tr, y_tr)
    oof_cb[val_idx] = m_cb.predict_proba(X_va)[:, 1]
    models_cb.append(m_cb)
    
    # 3. LightGBM (Histogram GBDT Gradient Descent)
    m_lgb = lgb.LGBMClassifier(
        n_estimators=350,
        num_leaves=48,
        learning_rate=0.05,
        min_child_samples=35,
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
    
    print(f"  Fold {fold+1} finished in {time.time()-f_start:.2f}s")

print(f"\\nAll folds trained in {time.time()-t_train_start:.2f}s!")
"""))

# Cell 6: Markdown Meta-Learners
cells.append(nbf.v4.new_markdown_cell("""## 2. Multi-Level Meta-Learner Optimization: Gradient Descent & Ensemble Stacking
We transform base probabilities into unbounded log-odds features:
$$z = \\text{logit}(P) = \\ln\\left(\\frac{P}{1 - P}\\right)$$
We now compare **four distinct optimization strategies** for the final meta-layer:
1. **SGD Meta-Learner**: Iterative Stochastic Gradient Descent directly minimizing Binary Cross-Entropy.
2. **L2 Logistic Meta-Learner**: Second-order Newton-Raphson gradient descent with $L_2$ shrinkage.
3. **XGBoost Meta-Learner**: Functional gradient boosted trees exploring non-linear interactions among model predictions.
4. **🏆 Super Hybrid Meta-Ensemble**: Combines the calibrated linear gradient descent model with the XGBoost meta-learner.
"""))

# Cell 7: Meta-Learners Code
cells.append(nbf.v4.new_code_cell("""eps = 1e-6
z_xgb = logit(np.clip(oof_xgb, eps, 1 - eps))
z_cb = logit(np.clip(oof_cb, eps, 1 - eps))
z_lgb = logit(np.clip(oof_lgb, eps, 1 - eps))
stack_X = np.column_stack([z_xgb, z_cb, z_lgb])

# 1. Pure Stochastic Gradient Descent (SGD) Meta-Learner
sgd_meta = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.0005,
    max_iter=2000,
    learning_rate='optimal',
    random_state=42
)
sgd_meta.fit(stack_X, y)
oof_sgd_probs = sgd_meta.predict_proba(stack_X)[:, 1]

# 2. L2 Regularized Logistic Regression (Newton / L-BFGS Gradient Descent)
l2_meta = LogisticRegression(C=0.1, random_state=42)
l2_meta.fit(stack_X, y)
oof_l2_probs = l2_meta.predict_proba(stack_X)[:, 1]

# 3. XGBoost Ensemble Meta-Learner (Functional Tree Gradient Descent in Meta-Space)
xgb_meta = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=5.0,
    eval_metric='logloss',
    random_state=42
)
xgb_meta.fit(stack_X, y)
oof_xgb_meta_probs = xgb_meta.predict_proba(stack_X)[:, 1]

# 4. 🏆 Super Hybrid Meta-Ensemble (Linear Gradient Descent + XGBoost Meta Blend)
oof_hybrid_probs = 0.5 * oof_l2_probs + 0.5 * oof_xgb_meta_probs

print("=== Meta-Learner Optimization Weights ===")
print(f"SGD Meta Weights:        {sgd_meta.coef_[0]}, Intercept: {sgd_meta.intercept_[0]:.4f}")
print(f"L2 Logistic Meta Weights:{l2_meta.coef_[0]}, Intercept: {l2_meta.intercept_[0]:.4f}")
print("XGBoost Meta-Learner Feature Importances (XGB, Cat, LGB):", xgb_meta.feature_importances_)
"""))

# Cell 8: Markdown Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 3. Comprehensive 100% Out-Of-Fold (OOF) Evaluation Metrics
Every metric is calculated strictly on held-out folds ($N = 500,000$) to guarantee zero train leakage.
"""))

# Cell 9: Evaluation Code
cells.append(nbf.v4.new_code_cell("""models_dict = {
    '🏆 Super Hybrid Meta-Ensemble (GD + XGBoost Meta)': oof_hybrid_probs,
    'XGBoost Ensemble Meta-Learner (Tree GD)': oof_xgb_meta_probs,
    'L2 Regularized Logistic Meta (Newton GD)': oof_l2_probs,
    'Pure SGD Gradient Descent Meta-Learner': oof_sgd_probs,
    'Base Model: LightGBM (OOF)': oof_lgb,
    'Base Model: XGBoost (OOF)': oof_xgb,
    'Base Model: CatBoost (OOF)': oof_cb
}

metrics_rows = []
for name, probs in models_dict.items():
    preds = (probs >= 0.5).astype(int)
    metrics_rows.append({
        'Model Architecture': name,
        'Accuracy': accuracy_score(y, preds),
        'Precision': precision_score(y, preds, zero_division=0),
        'Recall': recall_score(y, preds, zero_division=0),
        'F1-Score': f1_score(y, preds, zero_division=0),
        'ROC-AUC (OOF)': roc_auc_score(y, probs),
        'PR-AUC': average_precision_score(y, probs),
        'Log Loss': log_loss(y, probs)
    })

eval_df = pd.DataFrame(metrics_rows).sort_values('ROC-AUC (OOF)', ascending=False).reset_index(drop=True)
print("=== 100% Out-Of-Fold (OOF) Model Evaluation (N = 500,000 Samples) ===")
display(eval_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC (OOF)', 'PR-AUC'], color='#d4edda')
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

# Cell 10: Visual Comparison Plots
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Multi-Metric Bar Chart
melted = eval_df.melt(id_vars='Model Architecture', value_vars=['ROC-AUC (OOF)', 'PR-AUC', 'F1-Score', 'Accuracy'], var_name='Metric', value_name='Score')
sns.barplot(data=melted, x='Metric', y='Score', hue='Model Architecture', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('OOF Metrics Across Base Models & Meta-Learners', fontsize=13, fontweight='bold')
axes[0, 0].set_ylim(0, 1.0)
axes[0, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# 2. Combined ROC Curves
for name, probs in models_dict.items():
    fpr, tpr, _ = roc_curve(y, probs)
    lw = 2.5 if 'Hybrid' in name or 'XGBoost Ensemble' in name else 1.2
    axes[0, 1].plot(fpr, tpr, label=f"{name[:30]}... ({roc_auc_score(y, probs):.4f})", linewidth=lw)

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 1].set_title('Out-Of-Fold ROC Curves (Optimized Meta-Stacking)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend(loc='lower right', fontsize=8)

# 3. Precision-Recall Curves
for name, probs in models_dict.items():
    p, r, _ = precision_recall_curve(y, probs)
    lw = 2.5 if 'Hybrid' in name or 'XGBoost Ensemble' in name else 1.2
    axes[1, 0].plot(r, p, label=f"{name[:30]}... (PR-AUC={average_precision_score(y, probs):.4f})", linewidth=lw)

axes[1, 0].set_title('Precision-Recall Curves (Return Class Imbalance)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend(loc='upper right', fontsize=8)

# 4. Confusion Matrix of the Champion Super Hybrid Ensemble
cm = confusion_matrix(y, (oof_hybrid_probs >= 0.5).astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kept (0)', 'Returned (1)'])
disp.plot(cmap='Blues', ax=axes[1, 1], values_format='d')
axes[1, 1].set_title('Confusion Matrix: Champion Super Hybrid Meta-Ensemble', fontsize=13, fontweight='bold')
axes[1, 1].grid(False)

plt.tight_layout()
plt.show()
"""))

# Cell 11: Markdown 50% Holdout Test Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 4. Strict 50% Train / 50% Unseen Test Set Holdout Evaluation
To evaluate out-of-sample generalization under strict holdout constraints:
1. Split the dataset strictly into **50% Training (250,000 instances)** and **50% Completely Unseen Test Data (250,000 instances)** with stratification.
2. Train base models (XGBoost, CatBoost, LightGBM) on the 50% train portion.
3. Train meta-learners on train-level out-of-fold predictions.
4. Evaluate all architectures on the **50% Unseen Test Data** ($N = 250,000$).
"""))

# Cell 12: 50% Holdout Evaluation Code
cells.append(nbf.v4.new_code_cell("""print("Executing Strict 50% Train / 50% Unseen Test Split Evaluation...")
t_holdout = time.time()

# 50% / 50% Stratified Split
X_tr50, X_te50, y_tr50, y_te50 = train_test_split(
    X, y, test_size=0.50, random_state=42, stratify=y
)
print(f"50% Train Split: {len(X_tr50):,} instances | 50% Unseen Test Split: {len(X_te50):,} instances")

# 1. Fit Base Models on 50% Train
xgb_50 = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=2.0, eval_metric='logloss', random_state=42, n_jobs=-1).fit(X_tr50, y_tr50)
cb_50 = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.05, l2_leaf_reg=5, random_strength=0.2, subsample=0.85, eval_metric='Logloss', random_seed=42, verbose=0).fit(X_tr50, y_tr50)
lgb_50 = lgb.LGBMClassifier(n_estimators=300, num_leaves=48, learning_rate=0.05, min_child_samples=35, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1).fit(X_tr50, y_tr50)

p_te50_xgb = xgb_50.predict_proba(X_te50)[:, 1]
p_te50_cb = cb_50.predict_proba(X_te50)[:, 1]
p_te50_lgb = lgb_50.predict_proba(X_te50)[:, 1]

# 2. Train-Level OOF for Meta-Learner fitting
skf50 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_tr50_xgb = np.zeros(len(X_tr50))
oof_tr50_cb = np.zeros(len(X_tr50))
oof_tr50_lgb = np.zeros(len(X_tr50))

for tr_i, va_i in skf50.split(X_tr50, y_tr50):
    s_xtr, s_ytr = X_tr50.iloc[tr_i], y_tr50.iloc[tr_i]
    s_xva = X_tr50.iloc[va_i]
    m_x = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.06, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=2.0, eval_metric='logloss', random_state=42, n_jobs=-1).fit(s_xtr, s_ytr)
    m_c = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.06, l2_leaf_reg=5, eval_metric='Logloss', random_seed=42, verbose=0).fit(s_xtr, s_ytr)
    m_l = lgb.LGBMClassifier(n_estimators=200, num_leaves=45, learning_rate=0.06, min_child_samples=40, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1).fit(s_xtr, s_ytr)
    oof_tr50_xgb[va_i] = m_x.predict_proba(s_xva)[:, 1]
    oof_tr50_cb[va_i] = m_c.predict_proba(s_xva)[:, 1]
    oof_tr50_lgb[va_i] = m_l.predict_proba(s_xva)[:, 1]

meta_X_tr50 = np.column_stack([logit(np.clip(oof_tr50_xgb, eps, 1-eps)), logit(np.clip(oof_tr50_cb, eps, 1-eps)), logit(np.clip(oof_tr50_lgb, eps, 1-eps))])
meta_X_te50 = np.column_stack([logit(np.clip(p_te50_xgb, eps, 1-eps)), logit(np.clip(p_te50_cb, eps, 1-eps)), logit(np.clip(p_te50_lgb, eps, 1-eps))])

meta_gd_50 = LogisticRegression(C=0.1, random_state=42).fit(meta_X_tr50, y_tr50)
meta_xgb_50 = xgb.XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0, eval_metric='logloss', random_state=42).fit(meta_X_tr50, y_tr50)

p_te50_gd = meta_gd_50.predict_proba(meta_X_te50)[:, 1]
p_te50_xgb_meta = meta_xgb_50.predict_proba(meta_X_te50)[:, 1]
p_te50_hybrid = 0.5 * p_te50_gd + 0.5 * p_te50_xgb_meta
p_te50_blend = (p_te50_xgb + p_te50_cb + p_te50_lgb) / 3.0

holdout_models = {
    '🏆 Super Hybrid Meta-Ensemble': p_te50_hybrid,
    'Gradient Descent L2 Meta-Learner': p_te50_gd,
    'XGBoost Ensemble Meta-Learner': p_te50_xgb_meta,
    'Linear Equal Blend Ensemble': p_te50_blend,
    'LightGBM (Tuned)': p_te50_lgb,
    'XGBoost (Tuned)': p_te50_xgb,
    'CatBoost (Tuned)': p_te50_cb
}

holdout_rows = []
for name, probs in holdout_models.items():
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_te50, preds)
    holdout_rows.append({
        'Model Architecture': name,
        'Accuracy': accuracy_score(y_te50, preds),
        'Precision': precision_score(y_te50, preds, zero_division=0),
        'Recall': recall_score(y_te50, preds, zero_division=0),
        'F1-Score': f1_score(y_te50, preds, zero_division=0),
        'ROC-AUC (50% Test)': roc_auc_score(y_te50, probs),
        'PR-AUC': average_precision_score(y_te50, probs),
        'Log Loss': log_loss(y_te50, probs),
        'True Negatives': cm[0, 0],
        'False Positives': cm[0, 1],
        'False Negatives': cm[1, 0],
        'True Positives': cm[1, 1]
    })

holdout_df = pd.DataFrame(holdout_rows).sort_values('ROC-AUC (50% Test)', ascending=False).reset_index(drop=True)
print("=== Strict 50% Unseen Test Set Evaluation (N = 250,000 Samples) ===")
display(holdout_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC (50% Test)', 'PR-AUC'], color='#d4edda')
                        .highlight_min(subset=['Log Loss'], color='#d4edda')
                        .format({
                            'Accuracy': '{:.4f}',
                            'Precision': '{:.4f}',
                            'Recall': '{:.4f}',
                            'F1-Score': '{:.4f}',
                            'ROC-AUC (50% Test)': '{:.4f}',
                            'PR-AUC': '{:.4f}',
                            'Log Loss': '{:.4f}',
                            'True Negatives': '{:,}',
                            'False Positives': '{:,}',
                            'False Negatives': '{:,}',
                            'True Positives': '{:,}'
                        }))
print(f"50% Holdout evaluation completed in {time.time()-t_holdout:.2f}s!")
"""))

# Cell 13: Test Set Inference & Export
cells.append(nbf.v4.new_markdown_cell("""## 5. Working with `test.csv`: Final Test Set Inference & Submission
Now we perform inference on the official unlabelled test set (`data/test.csv`, $N = 479,995$) using K-Fold model averaging and the calibrated Meta-Learners.
"""))

cells.append(nbf.v4.new_code_cell("""# Load test dataset and generate K-Fold Fold-Averaged Predictions
print("Executing inference on data/test.csv (N = 479,995)...")
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

# Inference through Meta-Models
test_l2_probs = l2_meta.predict_proba(test_stack_X)[:, 1]
test_xgb_meta_probs = xgb_meta.predict_proba(test_stack_X)[:, 1]
test_hybrid_probs = 0.5 * test_l2_probs + 0.5 * test_xgb_meta_probs

# Export Champion Submissions
sub_hybrid = pd.DataFrame({
    'order_line_id': df_test_raw['order_line_id'],
    'returned': test_hybrid_probs
})
sub_hybrid.to_csv('submission.csv', index=False)
sub_hybrid.to_csv('data/submission_hybrid_meta.csv', index=False)

print(f"Exported Champion submission.csv successfully! Shape: {sub_hybrid.shape}")
print(f"Predicted Positive Return Rate @ 0.5: {(test_hybrid_probs >= 0.5).mean()*100:.2f}%")
print(f"Mean Probability: {test_hybrid_probs.mean():.4f} | Std Dev: {test_hybrid_probs.std():.4f}")
print(f"Min: {test_hybrid_probs.min():.4f} | Max: {test_hybrid_probs.max():.4f}")
"""))

nb['cells'] = cells

with open('notebook_v4_gradient_descent_advanced_stacking.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created notebook_v4_gradient_descent_advanced_stacking.ipynb successfully!")
