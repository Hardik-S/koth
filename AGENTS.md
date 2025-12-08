# Project Execution Plan for DS3000 KoTH

This repository currently contains Phase 2 materials for the King of the Hill chess evaluation project. Use this plan to drive it to a submission-ready Phase 3 package (executable notebook + final report) without fabricating results.

## Environment and Data
- Verify python-chess, datasets, scikit-learn, matplotlib, seaborn availability at start; install if missing.
- First choice: load the real Lichess KotH dataset via `datasets.load_dataset`. If unavailable, check for a local copy before resorting to clearly labeled synthetic data. Never hide missing data; annotate any placeholder metrics with `[MISSING: ...]`.

## Notebook Execution
- Run the existing Phase 2 notebook top-to-bottom, fixing failures minimally (dataset schema changes, parsing errors, scaling, split leaks). Keep game-level splits.
- After validation, build a clean Phase 3 notebook with sections: imports/env check, data loading/description, cleaning/preprocessing, EDA, modeling (logreg + MLP), evaluation/comparison, king-centralization analysis, feature importance, final summary/artifacts.

## Modeling and Analysis
- Logistic Regression: grid over C with L2, class_weight as needed; report accuracy/precision/recall/F1/ROC-AUC/confusion matrix.
- MLP: grid over hidden layers/alpha with early stopping; evaluate with same metrics. Compare both on identical test split.
- Perform EDA (class balance, feature distributions, king distance to center) and king-centralization subset analysis.
- Extract and interpret logistic regression coefficients for feature importance.

## Reporting
- Create a Markdown final report following instructor rubric: abstract, intro, background, methods, evaluation/results, interpretation, conclusion, rubric checklist. Only include metrics produced in this environment; mark missing items explicitly.
- Save key artifacts (models/metrics) and ensure notebook+report traceability: every reported number comes from executed code.

## Validation
- Before final delivery, confirm the Phase 3 notebook runs cleanly, cite any remaining limitations, and ensure rubric items are marked satisfied/partial/missing with reasons.
