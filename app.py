# app.py

import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import io

# Title
st.title("Structured Insights: Feedback Sentiment Analyzer")
st.markdown("Convert unstructured feedback into structured, analyzable sentiment data.")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'Feedback'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Feedback' not in df.columns:
        st.error("CSV must contain a 'Feedback' column.")
    else:
        # Preprocessing and Sentiment Analysis
        def get_sentiment(text):
            text = str(text).lower().strip()
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                return 'Positive', polarity
            elif polarity < 0:
                return 'Negative', polarity
            else:
                return 'Neutral', polarity

        sentiments = df['Feedback'].apply(get_sentiment)
        df[['Sentiment', 'Polarity Score']] = pd.DataFrame(sentiments.tolist(), index=df.index)

        # Display structured data
        st.subheader("Structured Sentiment Data")
        st.dataframe(df)

        # Visualization
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Download processed data
        output = io.BytesIO()
        df.to_csv(output, index=False)
        st.download_button(
            label="Download Structured CSV",
            data=output.getvalue(),
            file_name="structured_feedback.csv",
            mime="text/csv"
        )
