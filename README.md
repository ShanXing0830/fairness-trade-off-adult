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
  
- Both Logistic Regression and the PyTorch MLP achieve strong predictive performance on the Adult dataset (AUC around ~0.9).
- Under the default decision threshold, demographic disparity is observed across groups (non-zero SPD / DI < 1).
- Fairness mitigation (reweighing and threshold calibration) reduces disparity with a limited impact on accuracy.
## Accuracy–Fairness Tradeoff

![Accuracy–Fairness Tradeoff](figures/tradeoff.png)

## Run

```bash
pip install -r requirements.txt
python src/fairness_adult.py
