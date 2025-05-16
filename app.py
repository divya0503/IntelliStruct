import streamlit as st
import pandas as pd
import json
import docx
import fitz  # PyMuPDF for PDF reading
from io import StringIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Structured Insights", layout="wide")

st.title("📊 Structured Insights: Feedback Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload a file (CSV, TXT, PDF, DOCX, JSON, JSONL)", type=["csv", "txt", "pdf", "docx", "json", "jsonl"])

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def analyze_sentiment(text):
    from textblob import TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "json":
        df = pd.read_json(uploaded_file)
    elif file_type == "jsonl":
        df = pd.read_json(uploaded_file, lines=True)
    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
        df = pd.DataFrame([{"Feedback": line} for line in text.split('\n') if line.strip() != ""])
    elif file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
        df = pd.DataFrame([{"Feedback": line} for line in text.split('\n') if line.strip() != ""])
    elif file_type == "docx":
        text = extract_text_from_docx(uploaded_file)
        df = pd.DataFrame([{"Feedback": line} for line in text.split('\n') if line.strip() != ""])
    else:
        st.error("Unsupported file format!")
        st.stop()

    if "Feedback" not in df.columns:
        df.columns = [df.columns[0]]
        df.rename(columns={df.columns[0]: "Feedback"}, inplace=True)

    with st.expander("📄 Raw Feedback Data"):
        st.dataframe(df)

    df["Sentiment"] = df["Feedback"].apply(analyze_sentiment)

    with st.expander("📈 Structured Sentiment Data"):
        st.dataframe(df)

    st.download_button(
        label="⬇️ Download Structured CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="structured_feedback.csv",
        mime="text/csv"
    )


    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(sentiment_counts.set_index('Sentiment'))

    with col2:
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

