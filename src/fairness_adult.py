from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------
# Utils
# -------------------------
def reweighing_sample_weights(y, a):
    """
    Reweighing (Kamiran & Calders): w(y,a) = P(y)*P(a)/P(y,a)
    y: (n,) in {0,1}
    a: (n,) protected attribute in {0,1}  (e.g., 1=Male, 0=Female)
    """
    y = np.asarray(y).astype(int).ravel()
    a = np.asarray(a).astype(int).ravel()

    n = len(y)
    eps = 1e-12

    py = np.bincount(y, minlength=2) / (n + eps)
    pa = np.bincount(a, minlength=2) / (n + eps)

    joint = np.zeros((2, 2))
    for yi in (0, 1):
        for ai in (0, 1):
            joint[yi, ai] = np.mean((y == yi) & (a == ai))

    cell_w = np.zeros((2, 2))
    for yi in (0, 1):
        for ai in (0, 1):
            cell_w[yi, ai] = (py[yi] * pa[ai]) / (joint[yi, ai] + eps)

    w = np.array([cell_w[yi, ai] for yi, ai in zip(y, a)], dtype=float)
    return w / (np.mean(w) + eps)


def true_positive_rate(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    denom = (y_true == 1).sum()
    return (((y_pred == 1) & (y_true == 1)).sum() / denom) if denom > 0 else 0.0


def fairness_report(y_true, y_pred, male_mask, female_mask, title=""):
    p_male = y_pred[male_mask].mean()
    p_female = y_pred[female_mask].mean()
    spd = p_male - p_female
    di = p_female / (p_male + 1e-12)

    tpr_male = true_positive_rate(y_true[male_mask], y_pred[male_mask])
    tpr_female = true_positive_rate(y_true[female_mask], y_pred[female_mask])
    eod = tpr_male - tpr_female

    if title:
        print(f"\n=== {title} ===")
    print("Statistical Parity Difference:", spd)
    print("Equal Opportunity Difference:", eod)
    print("Disparate Impact (Female/Male):", di)
    return spd, eod, di


# Paths (robust savefig)
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)



# Load data
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame.copy()

for col in df.columns:
    if df[col].dtype.name == "category":
        df[col] = df[col].astype(str)

df["income"] = (df["class"] == ">50K").astype(int)
df = df.drop(columns=["class"])

X = df.drop(columns=["income"])
y = df["income"]

numeric_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_features = [col for col in X.columns if col not in numeric_features]

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# protected attribute + masks 
sex_train = X_train["sex"].values   
sex_test = X_test["sex"].values

a_train = (sex_train == "Male").astype(int)
a_test = (sex_test == "Male").astype(int)

male_mask = (sex_test == "Male")
female_mask = (sex_test == "Female")


# Logistic Regression (baseline)
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=2000))
])

logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
y_proba = logistic_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print("F1:", f1_score(y_test, y_pred))
fairness_report(y_test.values, y_pred, male_mask, female_mask, title="Logistic Regression Fairness")


# Logistic Regression (Reweighing)
w_train = reweighing_sample_weights(y_train.values, a_train)

logistic_rw = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=2000))
])


logistic_rw.fit(X_train, y_train, clf__sample_weight=w_train)

y_pred_rw = logistic_rw.predict(X_test)
y_proba_rw = logistic_rw.predict_proba(X_test)[:, 1]

print("\n=== Reweighing Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rw))
print("AUC:", roc_auc_score(y_test, y_proba_rw))
print("F1:", f1_score(y_test, y_pred_rw))
fairness_report(y_test.values, y_pred_rw, male_mask, female_mask, title="Reweighing Logistic Fairness")


# Threshold sweep (baseline LR)
thresholds = np.linspace(0.1, 0.9, 17)
print("\nThreshold Sweep (Logistic Baseline):")
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    acc_t = accuracy_score(y_test, y_pred_t)
    spd_t = y_pred_t[male_mask].mean() - y_pred_t[female_mask].mean()
    print(f"Threshold={t:.2f} | Acc={acc_t:.3f} | SPD={spd_t:.3f}")

print("\nThreshold Sweep (Logistic Reweighing):")
for t in thresholds:
    y_pred_t = (y_proba_rw >= t).astype(int)
    acc_t = accuracy_score(y_test, y_pred_t)
    spd_t = y_pred_t[male_mask].mean() - y_pred_t[female_mask].mean()
    print(f"Threshold={t:.2f} | Acc={acc_t:.3f} | SPD={spd_t:.3f}")


# MLP (PyTorch)
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

if hasattr(X_train_enc, "toarray"):
    X_train_enc = X_train_enc.toarray()
if hasattr(X_test_enc, "toarray"):
    X_test_enc = X_test_enc.toarray()

X_train_t = torch.tensor(X_train_enc, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test_enc, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(X_train_t.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    logits_test = model(X_test_t.to(device)).cpu().numpy().reshape(-1)
    y_proba_mlp = 1 / (1 + np.exp(-logits_test))

y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)

print("\n=== MLP Results ===")
print("Accuracy:", accuracy_score(y_test.values, y_pred_mlp))
print("AUC:", roc_auc_score(y_test.values, y_proba_mlp))
print("F1:", f1_score(y_test.values, y_pred_mlp))
fairness_report(y_test.values, y_pred_mlp, male_mask, female_mask, title="MLP Fairness")

print("\nThreshold Sweep (MLP):")
for t in thresholds:
    y_pred_t = (y_proba_mlp >= t).astype(int)
    acc_t = accuracy_score(y_test.values, y_pred_t)
    spd_t = y_pred_t[male_mask].mean() - y_pred_t[female_mask].mean()
    print(f"Threshold={t:.2f} | Acc={acc_t:.3f} | SPD={spd_t:.3f}")


# Tradeoff plot 
log_acc, log_spd = [], []
logrw_acc, logrw_spd = [], []
mlp_acc, mlp_spd = [], []

for t in thresholds:
    # baseline LR
    y_pred_log = (y_proba >= t).astype(int)
    log_acc.append(accuracy_score(y_test, y_pred_log))
    log_spd.append(y_pred_log[male_mask].mean() - y_pred_log[female_mask].mean())

    # reweighing LR
    y_pred_logrw = (y_proba_rw >= t).astype(int)
    logrw_acc.append(accuracy_score(y_test, y_pred_logrw))
    logrw_spd.append(y_pred_logrw[male_mask].mean() - y_pred_logrw[female_mask].mean())

    # MLP
    y_pred_m = (y_proba_mlp >= t).astype(int)
    mlp_acc.append(accuracy_score(y_test.values, y_pred_m))
    mlp_spd.append(y_pred_m[male_mask].mean() - y_pred_m[female_mask].mean())

plt.figure()
plt.plot(log_spd, log_acc, label="Logistic (Baseline)")
plt.plot(logrw_spd, logrw_acc, label="Logistic (Reweighing)")
plt.plot(mlp_spd, mlp_acc, label="MLP")
plt.xlabel("Statistical Parity Difference")
plt.ylabel("Accuracy")
plt.title("Accuracy–Fairness Tradeoff")
plt.legend()
plt.tight_layout()

plt.savefig(FIG_DIR / "tradeoff.png", dpi=200, bbox_inches="tight")

plt.show()
def pick_best_tradeoff_point(name, spd_list, acc_list, thresholds, lambda_spd=10.0):
    """
    Select the best accuracy–fairness tradeoff point.

    Score = lambda_spd * |SPD| + (1 - Accuracy)

    Parameters
    ----------
    name : str
        Model name for printing results.
    spd_list : list
        Statistical parity differences across thresholds.
    acc_list : list
        Accuracy values across thresholds.
    thresholds : list
        Threshold values used for evaluation.
    lambda_spd : float
        Weight controlling fairness importance.

    Returns
    -------
    best_index : int
        Index of the best tradeoff point.
    """

    spd_arr = np.asarray(spd_list)
    acc_arr = np.asarray(acc_list)
    thr_arr = np.asarray(thresholds)

    scores = lambda_spd * np.abs(spd_arr) + (1 - acc_arr)
    best_index = np.argmin(scores)

    print(f"\n=== Best tradeoff point: {name} ===")
    print(f"lambda_spd = {lambda_spd}")
    print(f"threshold  = {thr_arr[best_index]:.2f}")
    print(f"accuracy   = {acc_arr[best_index]:.4f}")
    print(f"SPD        = {spd_arr[best_index]:.4f}")
    print(f"score      = {scores[best_index]:.4f}")

    return best_index

pick_best_tradeoff_point("Logistic (Baseline)", log_spd, log_acc, thresholds)
pick_best_tradeoff_point("Logistic (Reweighing)", logrw_spd, logrw_acc, thresholds)
pick_best_tradeoff_point("MLP", mlp_spd, mlp_acc, thresholds)

plt.close("all")



 # Group-specific thresholds (Logistic baseline)
ts = np.linspace(0.1, 0.9, 33)
best = None

for t_m in ts:
    for t_f in ts:
        y_pred_g = np.zeros_like(y_proba, dtype=int)
        y_pred_g[male_mask] = (y_proba[male_mask] >= t_m).astype(int)
        y_pred_g[female_mask] = (y_proba[female_mask] >= t_f).astype(int)

        acc = accuracy_score(y_test, y_pred_g)
        f1 = f1_score(y_test, y_pred_g)
        spd = y_pred_g[male_mask].mean() - y_pred_g[female_mask].mean()

        score = abs(spd) * 10 + (1 - acc)
        if best is None or score < best[0]:
            best = (score, acc, f1, spd, t_m, t_f)

print("\nBest Group-Specific Thresholds (Logistic Baseline)")
print("Accuracy:", best[1])
print("F1:", best[2])
print("SPD:", best[3])
print("t_male:", best[4])
print("t_female:", best[5])


# Group-specific thresholds (MLP)
best = None
for t_m in ts:
    for t_f in ts:
        y_pred_g = np.zeros_like(y_proba_mlp, dtype=int)
        y_pred_g[male_mask] = (y_proba_mlp[male_mask] >= t_m).astype(int)
        y_pred_g[female_mask] = (y_proba_mlp[female_mask] >= t_f).astype(int)

        acc = accuracy_score(y_test.values, y_pred_g)
        f1 = f1_score(y_test.values, y_pred_g)
        spd = y_pred_g[male_mask].mean() - y_pred_g[female_mask].mean()

        score = abs(spd) * 10 + (1 - acc)
        if best is None or score < best[0]:
            best = (score, acc, f1, spd, t_m, t_f)

print("\nBest Group-Specific Thresholds (MLP)")
print("Accuracy:", best[1])
print("F1:", best[2])
print("SPD:", best[3])
print("t_male:", best[4])
print("t_female:", best[5])