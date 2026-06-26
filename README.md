#  AI Data Pipeline

A beginner-friendly AI data pipeline built with Python that fetches real weather data, cleans it, trains an AI model, and makes predictions.

## What It Does
- Fetches live weather data (temperature & rainfall) from a free API
- Cleans and processes the data automatically
- Trains an AI model to predict whether it will rain
- Saves the data to a CSV file

## Tech Stack
- **Python** — main programming language
- **Pandas** — data cleaning
- **Scikit-learn** — AI/ML model
- **Requests** — fetching data from the API
- **Git/GitHub** — version control

## How To Run It

**1. Clone the project:**
```bash
git clone https://github.com/KrithikaVijayakumar/ai-pipeline.git
cd ai-pipeline
```

**2. Install the libraries:**
```bash
pip install pandas requests scikit-learn
```

**3. Run the pipeline:**
```bash
python pipeline.py
```

## Output Example

✅ Got 168 rows of data!

✅ Data cleaned!

✅ Model trained!

 AI Prediction:

At 25°C, will it rain? No
✅ Data saved to weather_data.csv

 Pipeline complete!

 ## Author
Made by KrithikaVijayakumar 