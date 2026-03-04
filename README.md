# Accuracy–Fairness Tradeoff on the Adult Dataset

This project studies the tradeoff between predictive performance and algorithmic fairness using the Adult (Census Income) dataset.

Two models are compared:

- Logistic Regression (scikit-learn)
- MLP neural network (PyTorch)

Fairness is evaluated using:

- Statistical Parity Difference (SPD)
- Equal Opportunity Difference (EOD)
- Disparate Impact (DI)

We explore several fairness-aware strategies:

- Threshold sweeping to visualize the accuracy–fairness tradeoff
- Reweighing during training
- Group-specific threshold calibration

Both models achieve strong predictive performance (AUC ≈ 0.90+), but exhibit demographic disparities under a default threshold.  
Fairness mitigation methods can substantially reduce disparity with limited accuracy loss.

## Accuracy–Fairness Tradeoff

![Accuracy–Fairness Tradeoff](figures/tradeoff.png)

## Run

```bash
pip install -r requirements.txt
python src/fairness_adult.py