import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# === STEP 1: File Paths ===
MODEL_PATH = "job_title_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "jobs.csv"

# === STEP 2: Train Model if Not Exists ===
def train_and_save_model():
    print("🔧 Training model...")
    df = pd.read_csv(CSV_PATH)

    # Combine skills and job descriptions for better context
    df['text'] = df[['Skills', 'Job Description']].fillna('').agg(' '.join, axis=1)

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Job Title'])

    # Vectorize input
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])

    # Train classifier
    clf = LinearSVC()
    clf.fit(X, y)

    # Save models
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("✅ Model trained and saved to .pkl files.")

    return clf, vectorizer, label_encoder

# === STEP 3: Load Existing Model ===
def load_model():
    if not all(os.path.exists(f) for f in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]):
        return train_and_save_model()

    print("📦 Loading existing model from .pkl files...")
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return clf, vectorizer, label_encoder

# === STEP 4: Predict Job Title from Raw Post ===
def predict_job_title(raw_post):
    clf, vectorizer, label_encoder = load_model()
    raw_post = raw_post.lower()
    X = vectorizer.transform([raw_post])
    y_pred = clf.predict(X)
    return label_encoder.inverse_transform(y_pred)[0]

# === STEP 5: Main (Demo Input) ===
if __name__ == "__main__":
    # Replace with your own job post
    post = """
    HIRING!!! *Job Title:* Lead Generation Expert/Research Analyst *Location:* Remote *Job Type:* Contract
    We are seeking motivated individuals to join our team as Research Analysts. This role involves identifying and engaging industry professionals.
    Requirements include strong communication, research abilities, and the willingness to work independently.
    """

    try:
        prediction = predict_job_title(post)
        print("🔮 Predicted Job Title:", prediction)
    except Exception as e:
        print("❌ Error:", str(e))
