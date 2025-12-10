import streamlit as st
import pandas as pd
import yaml
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

df = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "train_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])

models_dir = Path(cfg["paths"]["models_dir"])
feature_cols = joblib.load(models_dir / "features.pkl")
model = joblib.load(models_dir / "model.pkl")
label_encoder = joblib.load(models_dir / "label_encoder.pkl")

test_start = pd.to_datetime(cfg["train"]["test_start"])
val_fraction = cfg["train"]["val_fraction"]

df_sorted = df.sort_values("Date")
train_df = df_sorted[df_sorted["Date"] < test_start]
test_df = df_sorted[df_sorted["Date"] >= test_start]

val_size = int(len(train_df) * val_fraction)
val_df = train_df.tail(val_size)
train_df2 = train_df.head(len(train_df) - val_size)


def section_dataset_distribution():
    st.subheader("Label Distribution")

    dist = df["Label"].value_counts().rename_axis("Label").reset_index(name="Count")
    st.table(dist)

    fig, ax = plt.subplots()
    dist.set_index("Label")["Count"].plot(kind="bar", ax=ax)
    st.pyplot(fig)


def section_feature_correlations():
    st.subheader("Feature Correlation Matrix")

    corr = df[feature_cols].corr()
    st.dataframe(corr)

    fig, ax = plt.subplots()
    c = ax.imshow(corr, cmap="coolwarm")
    fig.colorbar(c)
    st.pyplot(fig)


def section_model_performance():
    st.subheader("Model Performance on Validation/Test Sets")

    from sklearn.metrics import classification_report

    X_val = val_df[feature_cols].values
    y_val = label_encoder.transform(val_df["Label"])

    X_test = test_df[feature_cols].values
    y_test = label_encoder.transform(test_df["Label"])

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    st.write("Validation Report")
    st.text(classification_report(y_val, val_pred, target_names=label_encoder.classes_))

    st.write("Test Report")
    st.text(classification_report(y_test, test_pred, target_names=label_encoder.classes_))


st.title("Stock Model Diagnostics")

section_dataset_distribution()
section_feature_correlations()
section_model_performance()
