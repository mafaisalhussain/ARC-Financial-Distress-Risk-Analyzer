# ARC Financial Distress Model — Improvement Log

## Baseline (pre-improvement, 2026-06-07)

**Model:** LightGBM v3, 35 features, manually-tuned hyperparameters, fixed threshold 0.50
**Training data:** 110 companies (50 distressed, 60 healthy), pre-2020 train / 2020+ test

| Model               | Test AUC | F1     | Brier  |
|---------------------|----------|--------|--------|
| Altman Z (baseline) | 0.6702   | 0.4434 | —      |
| Logistic Regression | 0.7701   | 0.6248 | 0.1974 |
| Random Forest       | 0.8799   | 0.7471 | 0.1550 |
| Gradient Boosting   | 0.8865   | 0.7669 | 0.1274 |
| XGBoost             | 0.8810   | 0.7666 | 0.1300 |
| **LightGBM**        | **0.8760** | **0.7709** | **0.1342** |

**Threshold:** 0.50 (fixed)

---

## Step 1 — New XBRL features + Optuna tuning + Early stopping + Threshold optimisation (2026-06-07)

**Files changed:**
- `app/services/sec_service.py` — Added `InterestExpense`, `CapEx`, `LongTermDebt` to TAGS
- `app/services/feature_service.py` — Added 6 new features (35 → 41): `interest_coverage`, `interest_coverage_safe`, `interest_burden`, `fcf_ratio`, `fcf_positive`, `lt_debt_ratio`
- `train.py` — Updated TAGS + build_features + FEATURE_COLS; added 80-trial Optuna study, early stopping, PR-curve threshold optimisation
- `app/core/model_store.py` — Added `self.threshold: float` field
- `app/services/prediction_service.py` — Stores optimal threshold in metadata
- `requirements.txt` — Added `optuna>=3.0.0`

**Metrics before:** CV AUC — | Test AUC 0.8760 | F1 0.7709 | Brier 0.1342 | Threshold 0.50
**Metrics after:**  CV AUC 0.9902 | Test AUC 0.8949 | F1 0.7934 | Brier 0.1276 | Threshold 0.344
**Delta:** Test AUC +1.89 pp | F1 +2.25 pp | Brier −0.0066 (better calibration)
**Notes:**
- Optuna found: n_estimators=633, max_depth=7, lr=0.126, num_leaves=57 — substantially different from manual params (300/5/0.03/31)
- Early stopping triggered at 120 trees (vs fixed 300 before) — the higher learning rate converges faster
- Optimal threshold 0.344 (not 0.50): the model is better at lower probability cutoffs — prior 0.50 was missing distress cases
- Platt calibration reduces AUC slightly (0.9141 → 0.8949) but improves probability reliability (lower Brier)
- New XBRL fields (InterestExpense, CapEx, LongTermDebt) extracted from existing cached company JSONs — no re-fetch needed
- Model replaced in `ml_models/lightgbm.pkl` (new features stored; SHAP self-adapts via `model_store.feature_names`)
