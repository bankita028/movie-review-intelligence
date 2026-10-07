
import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


def predict_sentiment(review):
    review_tfidf = tfidf.transform([review])
    prediction = model.predict(review_tfidf)[0]
    probabilities = model.predict_proba(review_tfidf)[0]
    confidence = max(probabilities) * 100

    return prediction, confidence


st.set_page_config(
    page_title="Movie Review Intelligence",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Intelligence")

st.write(
    "An AI-powered sentiment analysis tool that predicts "
    "whether a movie review is positive or negative."
)

st.divider()

review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: The movie was brilliant and the performances were outstanding!",
    height=150
)

if st.button("🔍 Analyze Review", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a movie review first.")

    else:
        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if sentiment == "positive":
            st.success("🟢 POSITIVE")
        else:
            st.error("🔴 NEGATIVE")

        st.metric(
            "Prediction Probability",
            f"{confidence:.2f}%"
        )

        st.divider()

        st.caption(
            "Model: TF-IDF + Logistic Regression | "
            "Trained on 50,000 IMDb movie reviews"
        )
