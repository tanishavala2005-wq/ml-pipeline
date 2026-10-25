import joblib
import pandas as pd

from src.preprocess import load_config


def load_model():
    config = load_config()
    return joblib.load(config["model_path"])


def predict_churn(customer, model=None):
    if model is None:
        model = load_model()

    df = pd.DataFrame([customer])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    probability = model.predict_proba(df)[0][1]
    churn = int(model.predict(df)[0])

    return {"churn": churn, "probability": round(float(probability), 3)}


if __name__ == "__main__":
    risky_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    }

    safe_customer = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
    }

    print("Risky customer:", predict_churn(risky_customer))
    print("Safe customer :", predict_churn(safe_customer))
    