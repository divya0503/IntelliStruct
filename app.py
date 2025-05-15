import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Structured Insights: Feedback Sentiment Analyzer (Advanced)")

uploaded_file = st.file_uploader("Upload a CSV file with a column named 'Feedback'", type=["csv"])

# Load sentiment-analysis pipeline once (will download model if not cached)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

nlp = load_model()

def analyze_sentiment_transformers(text):
    try:
        result = nlp(text[:512])[0]  # limit to 512 tokens
        label = result['label']
        if label == 'NEGATIVE':
            return "Negative"
        elif label == 'POSITIVE':
            return "Positive"
        else:
            return "Neutral"
    except Exception as e:
        return "Neutral"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Feedback' not in df.columns:
        st.error("CSV file must contain a 'Feedback' column.")
    else:
        with st.spinner("Analyzing sentiments..."):
            df['Sentiment'] = df['Feedback'].astype(str).apply(analyze_sentiment_transformers)

        st.subheader("📄 Structured Feedback Data")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Structured Data", csv, "structured_feedback_advanced.csv", "text/csv")

        st.subheader("📈 Sentiment Distribution")

        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Count',
                         title='Sentiment Breakdown', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_bar, ax = plt.subplots()
        sns.barplot(data=sentiment_counts, x='Sentiment', y='Count', palette='Set2', ax=ax)
        ax.set_title('Sentiment Count')
        st.pyplot(fig_bar)
