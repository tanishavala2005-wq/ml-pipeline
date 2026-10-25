import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def build_preprocessor(X):
    cat_cols = [c for c in X.columns if c not in NUM_COLS]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])