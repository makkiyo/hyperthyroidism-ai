import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

file_path = r"C:\Users\PC\Downloads\thyroid+disease\allhyper.data"

df = pd.read_csv(
    file_path,
    header=None,
    na_values="?"
)

columns = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary",
    "psych", "TSH_measured", "TSH", "T3_measured", "T3",
    "TT4_measured", "TT4", "T4U_measured", "T4U",
    "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source", "target"
]

df.columns = columns

df = df[["TSH", "T3", "TT4", "target"]]

df = df.dropna()

df["target"] = df["target"].apply(
    lambda x: 1 if "hyper" in x.lower() else 0
)

X = df[["TSH", "T3", "TT4"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "model/hyperthyroid_model.pkl")
print("Model saved successfully!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

sample = [[0.15, 2.8, 16.0]]
prediction = model.predict(sample)

print(
    "Prediction:",
    "High risk of Hyperthyroidism" if prediction[0] == 1 else "Normal"
)