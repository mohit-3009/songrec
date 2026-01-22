from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

app = Flask(__name__)

# ================= LOAD DATA =================
data = pd.read_csv("songs_2000_2020_50k.csv")

data.drop_duplicates(subset=['Title', 'Artist'], inplace=True)

# Target variable
data['recommended'] = data['Popularity'].apply(
    lambda x: 1 if x >= 70 else 0
)

# Encode Genre
genre_encoder = LabelEncoder()
data['genre_encoded'] = genre_encoder.fit_transform(data['Genre'])

# Extract year
data['year'] = pd.to_datetime(
    data['Release Date'], errors='coerce'
).dt.year
data['year'].fillna(data['year'].median(), inplace=True)

# Features
features = ['genre_encoded', 'year', 'Duration']
X = data[features]
y = data['recommended']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= MODELS =================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    song_name = request.form['song_name'].strip().lower()

    # 🔍 Find the song
    matched_song = data[data['Title'].str.lower() == song_name]

    if matched_song.empty:
        return render_template(
            "result.html",
            error="Song not found in dataset. Please try another song."
        )

    # Get genre of entered song
    genre = matched_song.iloc[0]['Genre']
    genre_id = matched_song.iloc[0]['genre_encoded']

    # Get all songs of same genre
    genre_songs = data[data['genre_encoded'] == genre_id].copy()

    X_genre = genre_songs[features]
    X_genre = scaler.transform(X_genre)

    # Ensemble prediction
    rf_prob = rf.predict_proba(X_genre)[:, 1]
    xgb_prob = xgb.predict_proba(X_genre)[:, 1]

    genre_songs['ensemble_score'] = (rf_prob + xgb_prob) / 2

    # Top 6 songs
    top6 = genre_songs.sort_values(
        by='ensemble_score', ascending=False
    ).head(6)

    songs = top6[['Title', 'Artist', 'Album', 'year']].values.tolist()

    return render_template(
        "result.html",
        input_song=song_name.title(),
        genre=genre,
        songs=songs
    )

if __name__ == "__main__":
    app.run(debug=True)

