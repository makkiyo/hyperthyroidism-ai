import joblib

model = joblib.load("model/hyperthyroid_model.pkl")

def predict_hyperthyroidism(sample):
    pred = model.predict([sample])[0]
    return "High risk of Hyperthyroidism" if pred == 1 else "Normal"

sample = [0.15, 2.8, 16.0]
print("Example Prediction:")
print(f"Sample {sample} -> {predict_hyperthyroidism(sample)}\n")

print("Enter your own test values to predict:")
try:
    tsh = float(input("TSH Level: "))
    t3 = float(input("T3 Level: "))
    tt4 = float(input("TT4 Level: "))
    
    user_sample = [tsh, t3, tt4]
    print(f"Your Prediction: {predict_hyperthyroidism(user_sample)}")
except ValueError:
    print("Please enter valid numeric values.")