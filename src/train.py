import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocess import load_config, load_data, build_preprocessor


def main():
    config = load_config()

    # 1. Load data
    df = load_data(config["data_path"])
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 2. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    # 3. Build pipeline: preprocessing + model
    pipe = Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    # 4. Tune C with 5-fold cross-validation
    grid = GridSearchCV(pipe, {"model__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=5)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("Best C:", grid.best_params_["model__C"])

    # 5. Check on test data
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Recall   :", round(recall_score(y_test, pred), 3))
    print("F1       :", round(f1_score(y_test, pred), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_test, proba), 3))

    # 6. Save the model
    joblib.dump(best_model, config["model_path"])
    print("Model saved to", config["model_path"])


if __name__ == "__main__":
    main()
    