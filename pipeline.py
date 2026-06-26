import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# STEP 1: GET DATA (free weather API)
# ----------------------------
print("Fetching data...")

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 39.5,
    "longitude": -104.8,
    "hourly": "temperature_2m,precipitation"
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame({
    "time":        data["hourly"]["time"],
    "temperature": data["hourly"]["temperature_2m"],
    "rain":        data["hourly"]["precipitation"]
})

print(" Got", len(df), "rows of data!")
print(df.head())

# ----------------------------
# STEP 2: CLEAN THE DATA
# ----------------------------
print("\nCleaning data...")
df = df.dropna()  # remove any empty rows
print(" Data cleaned!")

# ----------------------------
# STEP 3: SIMPLE AI PREDICTION
# (will it rain? 1=yes, 0=no)
# ----------------------------
print("\nTraining AI model...")

df["will_rain"] = (df["rain"] > 0).astype(int)

X = df[["temperature"]]
y = df["will_rain"]

model = RandomForestClassifier()
model.fit(X, y)

print(" Model trained!")

# ----------------------------
# STEP 4: MAKE A PREDICTION
# ----------------------------
test_temp = [[25.0]]  # predict for 25 degrees Celsius
prediction = model.predict(test_temp)

print("\n AI Prediction:")
print(f"At 25°C, will it rain? {'Yes' if prediction[0] == 1 else 'No'}")

# ----------------------------
# STEP 5: SAVE THE DATA
# ----------------------------
df.to_csv("weather_data.csv", index=False)
print("\n Data saved to weather_data.csv")
print("\n Pipeline complete!")