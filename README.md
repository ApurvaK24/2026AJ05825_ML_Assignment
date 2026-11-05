# Dry Bean Variety Classification — ML Assignment 2

**Author:** Apurva Ajit Kulkarni (BITS ID 2026AJ05825)
**Course:** BITS WILP M.Tech (AIML) — Machine Learning, Assignment 2

## a. Problem statement

Identify a dry bean's variety (one of 7 types) purely from the geometry of its image
silhouette — 16 numeric shape measurements (areas, axis lengths, roundness/shape-factor
ratios), no color or texture information. Five classifiers are trained on the same dataset
and compared on six evaluation metrics, then wrapped in a Streamlit app that scores an
uploaded test CSV using the already-trained models.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (id 602)
- **Instances:** 13,611 — by far the largest of the three dataset variants used across
  this assignment's iterations.
- **Features:** 16 — all purely numeric, computer-vision-derived geometric measurements:
  `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`,
  `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, and four
  `ShapeFactor` ratios. There are no categorical/demographic columns at all.
- **Target:** `Class` — 7 bean varieties (`SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `HOROZ`,
  `SIRA`, `DERMASON`), imbalanced: `DERMASON` is largest (~26%), `BOMBAY` smallest (~3.8%).
- **Split:** stratified 80/20 train/test. The 20% test split is saved as
  [`test_data.csv`](test_data.csv) — this is the file used both for offline evaluation and
  for the Streamlit app demo.

## c. GitHub Repository Link


## d. Models used

Each model is a scikit-learn `Pipeline` (`StandardScaler` + classifier — no
`ColumnTransformer` is needed here since every feature is already numeric), tuned with
5-fold stratified cross-validation where applicable, and evaluated on the held-out 20%
test split. Full code: [`notebook/dry_bean_classification.ipynb`](notebook/dry_bean_classification.ipynb).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9207 | 0.9948 | 0.9352 | 0.9302 | 0.9324 | 0.9042 |
| Decision Tree | 0.8975 | 0.9734 | 0.9130 | 0.9102 | 0.9115 | 0.8761 |
| KNN | 0.9174 | 0.9870 | 0.9328 | 0.9279 | 0.9300 | 0.9001 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | 0.9243 | 0.9946 | 0.9372 | 0.9350 | 0.9361 | 0.9085 |

_(AUC/Precision/Recall/F1 are macro-averaged since this is a 7-class problem; MCC is the
multi-class generalized Matthews Correlation Coefficient. Numbers from `model/comparison_results.csv`,
produced by running `notebook/dry_bean_classification.ipynb` end-to-end with `random_state=42`.)_

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong (Accuracy 0.921, F1 0.932) despite being a linear model — most of the 16 geometric features (especially the four `ShapeFactor` ratios and `Eccentricity`) separate the 7 varieties close to linearly once standardized, so a complex decision boundary is not required. Best `C=100`. |
| Decision Tree | Weakest of the four non-Naive-Bayes models here (Accuracy 0.898, F1 0.912, best `criterion='entropy'`, `max_depth=10`, `min_samples_leaf=5`) — a single tree struggles to find the same smooth boundary a linear or margin-based model finds between visually similar varieties (e.g. `SIRA` vs `DERMASON`), even after tuning depth and split criterion. |
| KNN | Close behind the top two (Accuracy 0.917, F1 0.930, best `k=9`, `weights='distance'`) — with all 16 features genuinely continuous and standardized, Euclidean distance is meaningful here (unlike the one-hot-heavy datasets), so KNN finally performs close to its potential. |
| Naive Bayes | The standout result across all three dataset variants tried for this assignment: here Naive Bayes is competitive (Accuracy 0.898, F1 0.909) rather than collapsing. This confirms the earlier hypothesis from the Obesity/Student-Success datasets — Naive Bayes struggled there because of one-hot encoded categorical columns violating its Gaussian assumption, not because Naive Bayes is inherently weak. With every feature here genuinely continuous, it performs in line with the other four models. |
| Support Vector Machine | Best overall (Accuracy 0.924, F1 0.936, best `C=10`, `gamma='scale'`) — the RBF kernel finds the small amount of non-linearity between the closest varieties (`SIRA`/`DERMASON`, the two largest and most visually similar classes) that the linear model cannot, at the cost of being the slowest to train on 10,889 rows. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** — highest F1 (0.9361) and accuracy (0.9243) of the five, narrowly ahead of Logistic Regression (F1 0.9324). All five models land within a 4-point F1 band (0.909–0.936) — a sharp contrast with the Obesity and Student-Success datasets, where Naive Bayes was 30–50 points behind the rest. This dataset's all-numeric, well-separated geometric features are simply an easier, cleaner classification problem than survey/demographic data with many categorical columns. |

## Repository structure

```
dry-bean-classification/
├── app.py                     # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv              # held-out raw test split (features + target)
├── data/
│   └── dry_bean_raw.csv       # full raw dataset
├── model/                     # saved pipelines (StandardScaler + classifier) and metadata
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── label_encoder.joblib
│   ├── feature_columns.joblib
│   └── comparison_results.csv
└── notebook/
    └── dry_bean_classification.ipynb   # EDA, training, evaluation
```

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To regenerate `data/dry_bean_raw.csv`, `test_data.csv`, and the files in `model/`, open
`notebook/dry_bean_classification.ipynb` in Jupyter or VS Code and run all cells top to
bottom (requires `pip install jupyter` in addition to `requirements.txt`).

## Live Streamlit App Link
https://2026aj05825mlassignment.streamlit.app/
