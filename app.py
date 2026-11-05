"""
Dry Bean Variety Classifier - Streamlit demo app.

Loads five pre-trained scikit-learn pipelines (trained in
notebook/dry_bean_classification.ipynb on the UCI Dry Bean dataset) and scores a
user-uploaded test CSV. No retraining happens here - the app only loads saved
models and evaluates them.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")

MODEL_DIR = Path(__file__).parent / "model"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

CLASS_COLORS = {
    "SEKER": "#C77C02", "BARBUNYA": "#8E3B46", "BOMBAY": "#4A7A3D",
    "CALI": "#2F6690", "HOROZ": "#9C6B30", "SIRA": "#B08968", "DERMASON": "#6B4226",
}


@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    pipelines = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    return pipelines, label_encoder, feature_cols


def score_model(pipeline, X, y_true_encoded):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    auc = roc_auc_score(y_true_encoded, y_proba, multi_class="ovr", average="macro")
    metrics = {
        "Accuracy": accuracy_score(y_true_encoded, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true_encoded, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true_encoded, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true_encoded, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_encoded, y_pred),
    }
    return metrics, y_pred


def class_colors_for(labels):
    return [CLASS_COLORS.get(c, "#999999") for c in labels]


def donut_by_class(df, target_col):
    counts = df[target_col].value_counts()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    wedges, _ = ax.pie(
        counts.values, colors=class_colors_for(counts.index), startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="#1B1510"),
    )
    ax.legend(
        wedges, [f"{c} ({v})" for c, v in counts.items()],
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False,
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def scatter_by_class(df, target_col, x_col="Area", y_col="Roundness"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, sub in df.groupby(target_col):
        ax.scatter(sub[x_col], sub[y_col], s=14, alpha=0.6,
                   color=CLASS_COLORS.get(cls, "#999999"), label=cls)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(markerscale=2, fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def lollipop_chart(comparison_df, metric="F1"):
    ordered = comparison_df[metric].sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hlines(y=ordered.index, xmin=0, xmax=ordered.values, color="#C77C02", linewidth=2)
    ax.plot(ordered.values, ordered.index, "o", color="#C77C02", markersize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def feature_importance_chart(pipeline, model_name, feature_cols):
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"{model_name} - Gini feature importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
        title = f"{model_name} - mean |coefficient| across classes"
    else:
        return None, None
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_cols)[order], importances[order], color="#C77C02")
    ax.set_title(title)
    fig.tight_layout()
    return fig, title


pipelines, label_encoder, feature_cols = load_artifacts()
target_col = "Class"
n_classes = len(label_encoder.classes_)

with st.sidebar:
    st.header("🫘 Upload")
    uploaded_file = st.file_uploader(
        "Upload a test CSV",
        type=["csv"],
        help="Must contain all 16 feature columns plus the `Class` column.",
    )
    sample_path = Path(__file__).parent / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.title("🫘 Dry Bean Variety Classifier")

hero_cols = st.columns(4)
hero_cols[0].metric("Instances", "13,611")
hero_cols[1].metric("Features", "16 (all geometric)")
hero_cols[2].metric("Classes", "7 varieties")
hero_cols[3].metric("Problem type", "Multi-class")

with st.expander("ℹ️ Problem statement & dataset", expanded=False):
    st.markdown(
        "**Problem statement** — Identify a dry bean's variety (one of 7 types) purely from "
        "the geometry of its image silhouette: 16 numeric shape measurements (areas, axis "
        "lengths, roundness/shape-factor ratios), no color or texture information at all. Five "
        "classifiers are trained on the same dataset and compared on six evaluation metrics, "
        "then wrapped in this app, which scores an uploaded test CSV using the already-trained "
        "models (no retraining happens here)."
    )
    st.markdown(
        "**Dataset** — UCI *Dry Bean Dataset* (id 602): 13,611 instances, the largest of the "
        "three dataset variants used across this assignment's iterations. All 16 features are "
        "purely numeric, computer-vision-derived measurements (`Area`, `Perimeter`, "
        "`MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, "
        "`EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, and four "
        "`ShapeFactor` ratios) — no categorical/demographic columns. Target `Class` has 7 "
        "varieties, imbalanced: `DERMASON` is largest (~26%), `BOMBAY` smallest (~3.8%)."
    )

if uploaded_file is None:
    st.info("⬆️ Upload a CSV from the sidebar to see model predictions and evaluation metrics.")
    st.stop()

df = pd.read_csv(uploaded_file)

missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded file is missing required columns: {missing_cols}")
    st.stop()

X_uploaded = df[feature_cols]
unseen_labels = set(df[target_col].unique()) - set(label_encoder.classes_)
if unseen_labels:
    st.error(f"Uploaded target contains labels the models were not trained on: {unseen_labels}")
    st.stop()
y_uploaded_encoded = label_encoder.transform(df[target_col])

st.success(f"✅ Loaded {len(df)} rows from **{uploaded_file.name}**.")

model_choice = st.selectbox("Model", list(MODEL_FILES.keys()))

comparison_rows = []
predictions_by_model = {}
for name, pipeline in pipelines.items():
    m, y_pred = score_model(pipeline, X_uploaded, y_uploaded_encoded)
    m["Model"] = name
    comparison_rows.append(m)
    predictions_by_model[name] = y_pred

comparison_df = pd.DataFrame(comparison_rows).set_index("Model")[METRIC_NAMES]

view = st.segmented_control(
    "View", ["Overview", "Model Results", "Compare Models", "Feature Importance"],
    default="Overview",
)

if view == "Overview":
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Area vs. Roundness, by variety**")
        st.pyplot(scatter_by_class(df, target_col, "Area", "Roundness"))
    with right:
        st.markdown("**Class distribution**")
        st.pyplot(donut_by_class(df, target_col))
    st.markdown("**Preview**")
    st.dataframe(df.head(10), height=300)

elif view == "Model Results":
    st.subheader(f"Results — {model_choice}")
    chosen_pipeline = pipelines[model_choice]
    chosen_metrics = comparison_df.loc[model_choice]
    y_pred = predictions_by_model[model_choice]

    metric_cols = st.columns(6)
    for col, metric_name in zip(metric_cols, METRIC_NAMES):
        col.metric(metric_name, f"{chosen_metrics[metric_name]:.3f}")

    cm_tab, report_tab = st.tabs(["🔢 Confusion Matrix", "📋 Classification Report"])
    with cm_tab:
        cm = confusion_matrix(y_uploaded_encoded, y_pred, labels=range(n_classes))
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrBr",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    with report_tab:
        report = classification_report(
            y_uploaded_encoded, y_pred, target_names=label_encoder.classes_,
            zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), height=360, use_container_width=True)

    predictions_df = df.copy()
    predictions_df[f"predicted_{target_col}"] = label_encoder.inverse_transform(y_pred)
    st.download_button(
        f"⬇️ Download predictions ({model_choice})",
        data=predictions_df.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

elif view == "Compare Models":
    st.subheader("All five models on this uploaded file")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**F1 score, lowest to highest**")
        st.pyplot(lollipop_chart(comparison_df, "F1"))
    with right:
        st.markdown("**Accuracy, lowest to highest**")
        st.pyplot(lollipop_chart(comparison_df, "Accuracy"))
    st.dataframe(
        comparison_df.round(4).style.highlight_max(axis=0, color="#4a3521"),
        use_container_width=True,
    )

else:  # Feature Importance
    st.subheader(f"What drives {model_choice} predictions?")
    fig, title = feature_importance_chart(pipelines[model_choice], model_choice, feature_cols)
    if fig is None:
        st.info(
            f"{model_choice} doesn't expose a simple per-feature importance/coefficient "
            "(true for KNN, Naive Bayes, and RBF-kernel SVM). Pick Decision Tree or "
            "Logistic Regression to see which of the 16 geometric features matter most."
        )
    else:
        st.pyplot(fig)
