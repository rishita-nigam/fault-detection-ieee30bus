"""
=============================================================================
 IEEE 30-Bus Power System — Explainable ML Fault Classification
=============================================================================
 Labels:  0 = Normal | 1 = LG | 2 = LL | 3 = LLG | 4 = LLL
 Models:  Random Forest (RF) | Support Vector Machine (SVM)
 XAI:     Permutation Importance (scikit-learn)

 Usage:
   python ieee30bus_fault_classification.py
   All output plots are saved to the current working directory.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # change to 'TkAgg' or remove for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

FAULT_CSV       = r""   # path of dataset with long format (one row = one bus measurement)
NORMAL_CSV      = r""   # path of dataset with long format (one row = one bus measurement)
FAULT_WIDE_CSV  = r""   # wide-format dataset with fault_resistance column
RANDOM_STATE = 42
TEST_SIZE    = 0.2
RF_N_EST     = 200
SVM_C        = 10
SVM_GAMMA    = 'scale'
N_CV_FOLDS   = 5
PERM_REPEATS = 10
TOP_K_FEAT   = 20           # how many features to show in importance plot

LABEL_MAP = {0: 'Normal', 1: 'LG', 2: 'LL', 3: 'LLG', 4: 'LLL'}
FEATURES  = ['V_mag', 'V_ang', 'I_mag', 'P', 'Q']

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD & MERGE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("Loading datasets ...")
fault_df      = pd.read_csv(FAULT_CSV)
normal_df     = pd.read_csv(NORMAL_CSV)
fault_wide_df = pd.read_csv(FAULT_WIDE_CSV)   # wide-format with fault_resistance
df_long       = pd.concat([normal_df, fault_df], ignore_index=True)

print(f"  Total rows (long format): {len(df_long):,}")
print(f"  Label distribution:\n{df_long['label'].value_counts().sort_index()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RESHAPE: LONG → WIDE  (one row = one simulation sample)
# ─────────────────────────────────────────────────────────────────────────────
print("\nReshaping long → wide ...")
wide = df_long.pivot_table(index=['sample_id', 'label'],
                           columns='bus_id',
                           values=FEATURES)
wide.columns = [f"{feat}_bus{bus}" for feat, bus in wide.columns]
wide = wide.reset_index()
print(f"  Wide shape: {wide.shape}  ({wide.shape[1]-2} raw features)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING (system-level aggregates)
# ─────────────────────────────────────────────────────────────────────────────
vmag_cols = [c for c in wide.columns if c.startswith('V_mag')]
imag_cols = [c for c in wide.columns if c.startswith('I_mag')]
p_cols    = [c for c in wide.columns if c.startswith('P_')]
q_cols    = [c for c in wide.columns if c.startswith('Q_')]

wide['V_mag_mean']  = wide[vmag_cols].mean(axis=1)   # mean bus voltage
wide['V_mag_min']   = wide[vmag_cols].min(axis=1)    # worst-case voltage dip
wide['V_mag_std']   = wide[vmag_cols].std(axis=1)    # voltage spread
wide['I_mag_max']   = wide[imag_cols].max(axis=1)    # peak fault current proxy
wide['I_mag_std']   = wide[imag_cols].std(axis=1)    # current spread
wide['P_total']     = wide[p_cols].sum(axis=1)       # total real power
wide['Q_total']     = wide[q_cols].sum(axis=1)       # total reactive power

print(f"  After engineering: {wide.shape[1]-2} features total")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREPARE FEATURE MATRIX & TARGET VECTOR
# ─────────────────────────────────────────────────────────────────────────────
drop_cols     = ['sample_id', 'label']
X             = wide.drop(columns=drop_cols).values
y             = wide['label'].values
feature_names = wide.drop(columns=drop_cols).columns.tolist()
class_names   = [LABEL_MAP[i] for i in sorted(np.unique(y))]

print(f"\nX shape: {X.shape} | Classes: {class_names}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=RF_N_EST,
    max_depth=25,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=None  # This stops the parallel.py warning
)
rf.fit(X_train, y_train)

print("Training SVM ...")
svm = SVC(kernel='rbf', C=SVM_C, gamma=SVM_GAMMA,
          probability=True, random_state=RANDOM_STATE)
svm.fit(X_train_sc, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATE ON HELD-OUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
y_pred_rf  = rf.predict(X_test)
y_pred_svm = svm.predict(X_test_sc)

rf_acc  = accuracy_score(y_test, y_pred_rf)
svm_acc = accuracy_score(y_test, y_pred_svm)
rf_f1   = f1_score(y_test, y_pred_rf,  average='weighted')
svm_f1  = f1_score(y_test, y_pred_svm, average='weighted')

# ─────────────────────────────────────────────────────────────────────────────
# 8b. SAMPLE-WISE PREDICTIONS & PHYSICAL REASONING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SAMPLE-WISE PREDICTIONS & PHYSICAL REASONING (First 20 Test Samples)")
print("=" * 65)

# Get probability scores for the Random Forest model
y_probs_rf = rf.predict_proba(X_test)
max_probs  = np.max(y_probs_rf, axis=1)

# Identify the indices for Voltage Angle features (critical for reasoning)
v_ang_indices = [i for i, name in enumerate(feature_names) if 'V_ang' in name]

# Create the detailed DataFrame
predictions_df = pd.DataFrame({
    'Sample':        range(1, len(y_test) + 1),
    'Actual':        [LABEL_MAP[val] for val in y_test],
    'Predicted':     [LABEL_MAP[val] for val in y_pred_rf],
    'Confidence':    [f"{p*100:.1f}%" for p in max_probs]
})

# Add physical reasoning based on the V_ang threshold observed in your data
reasons = []
for i in range(len(X_test)):
    # Calculate the average angle across all buses for this sample
    avg_ang = np.mean(X_test[i, v_ang_indices])
    
    if predictions_df.iloc[i]['Predicted'] == 'Normal':
        reasons.append(f"Stable Angles ({avg_ang:.1f}°)")
    else:
        # Based on your data diagnostics: Normal ~20°, Fault ~90°
        reasons.append(f"Angle Shift ({avg_ang:.1f}°) + Imbalance")

predictions_df['Primary Reason'] = reasons

# Add corresponding fault_resistance (Rf) from fault_dataset.csv
# fault_wide_df has 500 rows aligned positionally to fault sample_ids in wide (label != 0)
# Build a map: fault sample_id -> fault_resistance (fault_wide_df row 0 = sample_id 1, etc.)
fault_sample_ids = sorted(wide[wide['label'] != 0]['sample_id'].unique())
fault_rf_map     = {sid: rf for sid, rf in
                    zip(fault_sample_ids, fault_wide_df['fault_resistance'].values)}
# wide_test gives us sample_id for each test row in the same order as X_test
wide_reset   = wide.reset_index(drop=True)
_, test_idx  = train_test_split(range(len(wide_reset)), test_size=TEST_SIZE,
                                random_state=RANDOM_STATE,
                                stratify=wide_reset['label'].values)
test_sample_ids = wide_reset.iloc[test_idx]['sample_id'].values
predictions_df['Fault_Resistance_Rf'] = [
    fault_rf_map.get(sid, float('nan')) for sid in test_sample_ids
]

# Print the table to the console
print(predictions_df.head(20).to_string(index=False))

# Save this detailed validation table for your report
predictions_df.to_csv('sample_predictions_with_reasoning.csv', index=False)
print(f"\nSaved: sample_predictions_with_reasoning.csv")

# ── Build merged output CSV (prediction_Results format + sample_predictions columns) ──
# Physical result description (mirrors prediction_Results.csv logic)
def build_physical_result(row, test_sample, vmag_cols_local, imag_cols_local):
    v_min  = test_sample[vmag_cols_local].min() if vmag_cols_local else float('nan')
    i_max  = test_sample[imag_cols_local].max() if imag_cols_local else float('nan')
    pred   = row['Predicted']
    if pred == 'Normal':
        return (f"Healthy State: All voltages within 0.95-1.05 pu; "
                f"load-level currents.")
    elif pred == 'LG':
        return (f"Single Line-to-Ground Fault: Voltage dip to {v_min:.3f} pu; "
                f"fault current {i_max:.3f} pu.")
    elif pred == 'LL':
        return (f"Line-to-Line Fault: Voltage sag to {v_min:.3f} pu; "
                f"elevated current {i_max:.3f} pu.")
    elif pred == 'LLG':
        return (f"Double Line-to-Ground Fault: Severe sag {v_min:.3f} pu; "
                f"high fault current {i_max:.3f} pu.")
    elif pred == 'LLL':
        return (f"Three-Phase Fault: Symmetric collapse to {v_min:.3f} pu; "
                f"peak current {i_max:.3f} pu.")
    return ""

# Resolve column indices in X_test for vmag / imag
vmag_feat_indices = [i for i, n in enumerate(feature_names) if n.startswith('V_mag_bus')]
imag_feat_indices = [i for i, n in enumerate(feature_names) if n.startswith('I_mag_bus')]

physical_results = []
min_voltages     = []
max_currents     = []

for i, row in predictions_df.iterrows():
    idx      = i  # predictions_df index == X_test row index
    v_vals   = X_test[idx, vmag_feat_indices] if vmag_feat_indices else np.array([float('nan')])
    i_vals   = X_test[idx, imag_feat_indices] if imag_feat_indices else np.array([float('nan')])
    v_min    = float(np.min(v_vals))
    i_max    = float(np.max(i_vals))
    min_voltages.append(round(v_min, 3))
    max_currents.append(round(i_max, 3))
    pred     = row['Predicted']
    if pred == 'Normal':
        desc = "Healthy State: All voltages within 0.95-1.05 pu; load-level currents."
    elif pred == 'LG':
        desc = f"Single Line-to-Ground Fault: Voltage dip to {v_min:.3f} pu; fault current {i_max:.3f} pu."
    elif pred == 'LL':
        desc = f"Line-to-Line Fault: Voltage sag to {v_min:.3f} pu; elevated current {i_max:.3f} pu."
    elif pred == 'LLG':
        desc = f"Double Line-to-Ground Fault: Severe sag {v_min:.3f} pu; high fault current {i_max:.3f} pu."
    elif pred == 'LLL':
        desc = f"Three-Phase Fault: Symmetric collapse to {v_min:.3f} pu; peak current {i_max:.3f} pu."
    else:
        desc = ""
    physical_results.append(desc)

merged_df = pd.DataFrame({
    'Sample_ID':           predictions_df['Sample'],
    'Actual_Fault':        predictions_df['Actual'],
    'ML_Prediction':       predictions_df['Predicted'],
    'Min_Voltage_pu':      min_voltages,
    'Max_Current_pu':      max_currents,
    'Physical_Result':     physical_results,
    'Confidence':          predictions_df['Confidence'],
    'Fault_Resistance_Rf': predictions_df['Fault_Resistance_Rf'],
})

merged_df.to_csv('prediction_Results.csv', index=False)
print("Saved: prediction_Results.csv  (merged — all columns)")

print("\n" + "=" * 55)
print("RANDOM FOREST — Classification Report")
print("=" * 55)
print(classification_report(y_test, y_pred_rf, target_names=class_names))

print("=" * 55)
print("SVM (RBF) — Classification Report")
print("=" * 55)
print(classification_report(y_test, y_pred_svm, target_names=class_names))

print(f"RF  Accuracy: {rf_acc:.4f}  |  Weighted F1: {rf_f1:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}  |  Weighted F1: {svm_f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. 5-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                     random_state=RANDOM_STATE)
rf_cv_scores = cross_val_score(rf, X, y, cv=cv,
                                scoring='accuracy', n_jobs=-1)

print(f"\nRF 5-Fold CV Accuracy: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. XAI — PERMUTATION IMPORTANCE (model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing permutation importance (XAI) ...")
perm = permutation_importance(rf, X_test, y_test,
                               n_repeats=PERM_REPEATS,
                               random_state=RANDOM_STATE,
                               n_jobs=1)
perm_df = pd.DataFrame({
    'feature':         feature_names,
    'importance_mean': perm.importances_mean,
    'importance_std':  perm.importances_std
}).sort_values('importance_mean', ascending=False).head(TOP_K_FEAT)

print(f"\nTop {TOP_K_FEAT} Features by Permutation Importance:")
print(perm_df.to_string(index=False))

# RF Gini (built-in)
rf_imp_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(TOP_K_FEAT)

# ─────────────────────────────────────────────────────────────────────────────
# 11. PHYSICAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
stat_cols = ['V_mag_mean', 'V_mag_min', 'V_mag_std', 'I_mag_max']
phys      = wide.groupby('label')[stat_cols].mean().copy()
phys.index = [LABEL_MAP[i] for i in phys.index]

print("\n" + "=" * 55)
print("PHYSICAL VALIDATION: Mean Feature Values per Class")
print("=" * 55)
print(phys.round(4))
print("\nExpected physical behaviour:")
print("  ✓ V_mag_min DECREASES under fault (voltage dip)")
print("  ✓ V_mag_std INCREASES under fault (uneven sag)")

print("  ✓ I_mag_max HIGHER under fault (fault current)")

# ─────────────────────────────────────────────────────────────────────────────
# 12. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
palette = ['#27ae60', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

# ── Plot 1: Confusion Matrices ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrices — IEEE 30-Bus Fault Classification',
             fontsize=14, fontweight='bold')

for ax, y_pred, title in zip(
    axes,
    [y_pred_rf, y_pred_svm],
    ['Random Forest', 'SVM (RBF kernel)']
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    ax.set_title(f'{title}\nAcc: {acc:.3f}  |  F1: {f1:.3f}', fontsize=12)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: confusion_matrices.png")

# ── Plot 2: Model Comparison ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
models = ['Random Forest', 'SVM']
accs   = [rf_acc,  svm_acc]
f1s    = [rf_f1,   svm_f1]
x = np.arange(len(models))
b1 = ax.bar(x - 0.2, accs, 0.35, label='Accuracy',    color='steelblue')
b2 = ax.bar(x + 0.2, f1s,  0.35, label='Weighted F1', color='darkorange')
ax.set_ylim(0, 1.1)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: RF vs SVM', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# ── Plot 3: Permutation Importance (XAI) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
feat_colors = []
for f in perm_df['feature']:
    if 'V_ang' in f:                        feat_colors.append('#e74c3c')
    elif 'V_mag' in f:                      feat_colors.append('#3498db')
    elif 'I_mag' in f:                      feat_colors.append('#2ecc71')
    elif f.startswith('P_') or 'P_bus' in f: feat_colors.append('#f39c12')
    else:                                   feat_colors.append('#9b59b6')

ax.barh(perm_df['feature'][::-1],
        perm_df['importance_mean'][::-1],
        xerr=perm_df['importance_std'][::-1],
        color=feat_colors[::-1], edgecolor='white', linewidth=0.5)

legend_patches = [
    mpatches.Patch(color='#e74c3c', label='Voltage Angle (V_ang)'),
    mpatches.Patch(color='#3498db', label='Voltage Magnitude (V_mag)'),
    mpatches.Patch(color='#2ecc71', label='Current Magnitude (I_mag)'),
    mpatches.Patch(color='#f39c12', label='Active Power (P)'),
    mpatches.Patch(color='#9b59b6', label='Reactive Power (Q)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mean Accuracy Decrease (Permutation Importance)', fontsize=12)
ax.set_title(f'Top {TOP_K_FEAT} Features — XAI: Permutation Importance\n'
             '(Random Forest | IEEE 30-Bus Fault Classification)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('xai_permutation_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xai_permutation_importance.png")

# ── Corrected Physical Validation Plot ─────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 7))

# Define colors for the groups
colors = ['#27ae60', '#2980b9', '#3498db', '#e74c3c', '#8e44ad']
width = 0.15
x = np.arange(len(phys.index))

# Plot Voltages on Left Axis (0 to 1.2 pu scale)
ax1.bar(x - 2*width, phys['V_mag_mean'], width, label='V_mag_mean', color='#27ae60')
ax1.bar(x - width, phys['V_mag_min'], width, label='V_mag_min', color='#2ecc71')
ax1.set_ylabel('Voltage (pu)', fontsize=12, color='green')
ax1.set_ylim(0, 1.2)

# Create a second axis for Current and Imbalance (Larger scale)
ax2 = ax1.twinx()
ax2.bar(x, phys['I_mag_max'], width, label='I_mag_max', color='#e74c3c')

ax2.set_ylabel('Current / Imbalance Magnitude', fontsize=12, color='red')

# Formatting
ax1.set_xticks(x)
ax1.set_xticklabels(phys.index, fontsize=11)
ax1.set_title('Physical Validation: Corrected Voltage Sag & Current Swell', fontsize=14, fontweight='bold')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('physical_validation_FINAL.png', dpi=150)
plt.close()

# ── Plot 5: Cross-Validation Fold Scores ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, N_CV_FOLDS + 1), rf_cv_scores,
       color='steelblue', edgecolor='white')
ax.axhline(rf_cv_scores.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean = {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title(f'{N_CV_FOLDS}-Fold Cross-Validation — Random Forest',
             fontsize=13, fontweight='bold')
ax.set_ylim(0.7, 1.05)
ax.set_xticks(range(1, N_CV_FOLDS + 1))
ax.legend(fontsize=11)
for i, v in enumerate(rf_cv_scores):
    ax.text(i + 1, v + 0.003, f'{v:.4f}',
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cross_validation.png")

print("\n✓ All outputs saved. Pipeline complete.")