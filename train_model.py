import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("crop_prices.csv")

# Label Encoding
encoder = LabelEncoder()

df["Crop"] = encoder.fit_transform(df["Crop"])
joblib.dump(encoder, "label_encoder.pkl")

state_encoder = LabelEncoder()
df["State"] = state_encoder.fit_transform(df["State"])
joblib.dump(state_encoder, "state_encoder.pkl")

X = df[["Crop","State","Month","Rainfall","Temperature","Demand"]]
y = df["Price"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler,"scaler.pkl")

# Model
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X,y)

joblib.dump(model,"model.pkl")

print("Model Saved Successfully")
