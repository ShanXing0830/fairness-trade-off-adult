from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

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
                 ]              )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print("Train:", X_train.shape)
print("Test:", X_test.shape)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

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


import numpy as np

sex_test = X_test["sex"].values

male_mask = (sex_test == "Male")
female_mask = (sex_test == "Female")

# Statistical Parity
p_male = y_pred[male_mask].mean()
p_female = y_pred[female_mask].mean()

spd = p_male - p_female

# Equal Opportunity (TPR difference)
def true_positive_rate(y_true, y_pred):
    return ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()

tpr_male = true_positive_rate(y_test.values[male_mask], y_pred[male_mask])
tpr_female = true_positive_rate(y_test.values[female_mask], y_pred[female_mask])

eod = tpr_male - tpr_female

print("Statistical Parity Difference:", spd)
print("Equal Opportunity Difference:", eod)
di = p_female / (p_male + 1e-12)
print("Disparate Impact (Female/Male):", di)


thresholds = np.linspace(0.1, 0.9, 17)

print("\nThreshold Sweep:")
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)

    acc_t = accuracy_score(y_test, y_pred_t)

    p_m = y_pred_t[male_mask].mean()
    p_f = y_pred_t[female_mask].mean()
    spd_t = p_m - p_f

    print(f"Threshold={t:.2f} | Acc={acc_t:.3f} | SPD={spd_t:.3f}")


    import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

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

print("\nMLP Results:")
print("Accuracy:", accuracy_score(y_test.values, y_pred_mlp))
print("AUC:", roc_auc_score(y_test.values, y_proba_mlp))
print("F1:", f1_score(y_test.values, y_pred_mlp))

sex_test = X_test["sex"].values
male_mask = (sex_test == "Male")
female_mask = (sex_test == "Female")

p_male = y_pred_mlp[male_mask].mean()
p_female = y_pred_mlp[female_mask].mean()
spd_mlp = p_male - p_female

def true_positive_rate(y_true, y_pred):
    return ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()

tpr_male = true_positive_rate(y_test.values[male_mask], y_pred_mlp[male_mask])
tpr_female = true_positive_rate(y_test.values[female_mask], y_pred_mlp[female_mask])
eod_mlp = tpr_male - tpr_female

print("Statistical Parity Difference:", spd_mlp)
print("Equal Opportunity Difference:", eod_mlp)
di_mlp = p_female / (p_male + 1e-12)
print("Disparate Impact (Female/Male):", di_mlp)

thresholds = np.linspace(0.1, 0.9, 17)
print("\nMLP Threshold Sweep:")
for t in thresholds:
    y_pred_t = (y_proba_mlp >= t).astype(int)
    acc_t = accuracy_score(y_test.values, y_pred_t)
    spd_t = y_pred_t[male_mask].mean() - y_pred_t[female_mask].mean()
    print(f"Threshold={t:.2f} | Acc={acc_t:.3f} | SPD={spd_t:.3f}")


    import matplotlib.pyplot as plt

log_acc = []
log_spd = []

mlp_acc = []
mlp_spd = []

thresholds = np.linspace(0.1, 0.9, 17)

for t in thresholds:
    y_pred_log = (y_proba >= t).astype(int)
    log_acc.append(accuracy_score(y_test, y_pred_log))
    log_spd.append(y_pred_log[male_mask].mean() - y_pred_log[female_mask].mean())

    y_pred_m = (y_proba_mlp >= t).astype(int)
    mlp_acc.append(accuracy_score(y_test.values, y_pred_m))
    mlp_spd.append(y_pred_m[male_mask].mean() - y_pred_m[female_mask].mean())


plt.savefig("figures/tradeoff.png", dpi=200, bbox_inches="tight")
plt.plot(log_spd, log_acc)
plt.plot(mlp_spd, mlp_acc)
plt.xlabel("Statistical Parity Difference")
plt.ylabel("Accuracy")
plt.title("Accuracy–Fairness Tradeoff")
plt.show()


import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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

print("\nBest Group-Specific Thresholds (Logistic)")
print("Accuracy:", best[1])
print("F1:", best[2])
print("SPD:", best[3])
print("t_male:", best[4])
print("t_female:", best[5])


ts = np.linspace(0.1, 0.9, 33)

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