import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from skforecast.recursive import ForecasterRecursive
import streamlit as st

# Load data
features = [
    "Tempo", "Loudness", "Energy", "Danceability",
    "Positiveness", "Acousticness",
    "emotion_encoded", "Key_encoded", "Explicit_encoded"
]

df = pd.read_excel("forecast_ready_yearly_weightedyeni.xlsx")
df["Release Date"] = pd.to_datetime(df["Release Date"], format='%Y')
df.set_index("Release Date", inplace=True)

genre_df = pd.read_excel("genre.xlsx")
genre_names = genre_df["Genre"]
genre_df = genre_df[features]

# Streamlit UI
st.title("🎵 Genre Predictor")
year = st.number_input("Enter target year (e.g. 2025):", min_value=2025, max_value=2100, value=2025)

# Prediction logic
def predict_features(target_year):
    predicted = {}
    last_year = df.index.year.max()
    steps = target_year - last_year
    if steps <= 0:
        return None

    for feature in features:
        series = df[feature]
        model = ForecasterRecursive(
            regressor=RandomForestRegressor(n_estimators=100, random_state=42),
            lags=10
        )
        model.fit(y=series)
        preds = model.predict(steps=steps)
        predicted[feature] = round(preds.iloc[-1], 6)
    
    return predicted

# Plot radar
def plot_radar(pred_dict, genre_vals, genre_name, year):
    labels = list(pred_dict.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    pred_values = list(pred_dict.values()) + [list(pred_dict.values())[0]]
    genre_values = genre_vals.tolist() + [genre_vals.tolist()[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, pred_values, color="red", linewidth=2, label="Predicted")
    ax.fill(angles, pred_values, color="red", alpha=0.25)
    ax.plot(angles, genre_values, color="blue", linewidth=2, label=genre_name)
    ax.fill(angles, genre_values, color="blue", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"{year} Feature Prediction vs Closest Genre", fontsize=11)
    ax.legend(loc="upper right")
    st.pyplot(fig)

# Run prediction
predicted = predict_features(year)
if predicted:
    pred_vec = np.array(list(predicted.values())).reshape(1, -1)
    genre_matrix = genre_df.to_numpy()
    similarities = cosine_similarity(genre_matrix, pred_vec).flatten()
    closest_index = np.argmax(similarities)
    closest_genre = genre_names.iloc[closest_index]
    genre_row = genre_df.iloc[closest_index]

    st.success(f"Predicted closest genre for {year}: {closest_genre}")
    plot_radar(predicted, genre_row, closest_genre, year)
else:
    st.error("Please enter a year after the latest in dataset.")
