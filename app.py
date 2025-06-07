import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BartForConditionalGeneration, BartTokenizer
import torch
import textstat
import matplotlib.pyplot as plt
import base64
import wikipedia
import requests
from io import StringIO
from PyPDF2 import PdfReader
import docx2txt
import transformers
print(transformers.__version__)


#  First Streamlit command - compulsary
st.set_page_config(page_title="🧠 Text Summarizer AI", layout="centered")

# Loading BART custom model
@st.cache_resource
def load_model():
    model = BartForConditionalGeneration.from_pretrained("model")
    tokenizer = BartTokenizer.from_pretrained("model")
    return model, tokenizer

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

model, tokenizer = load_model()
sentiment_pipeline = load_sentiment_pipeline()

# CSS Styling
st.markdown("""
    <style>
        .big-title {
            font-size: 2.5em; font-weight: 900;
            background: linear-gradient(to right, #1f4037, #99f2c8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .compact-summary {
            background-color: #f9f9f9;
            padding: 1em;
            border-left: 6px solid #4CAF50;
            border-radius: 10px;
            font-size: 0.95em;
            color: #333;
            line-height: 1.5em;
            margin-top: 1em;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        }
        .section-title {
            font-size: 1.3em; font-weight: 700; margin-top: 1.2em;
        }
        .advanced-box {
            background: #f1f8e9;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            box-shadow: 0 0 6px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Project Title
st.markdown('<div class="big-title">🤖 Text Summarizer using NLP </div>', unsafe_allow_html=True)
st.markdown("Paste a paragraph, upload a document, or fetch an article to get an AI-generated summary.")

# Input Options
text_input = st.text_area("📄 Enter your text manually (optional):", height=250)

uploaded_file = st.file_uploader("📤 Upload a file (PDF, TXT, DOCX)", type=['pdf', 'txt', 'docx'])

def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        if file_type == "pdf":
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_type == "txt":
            return StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif file_type == "docx":
            return docx2txt.process(uploaded_file)
    return None

file_text = extract_text_from_file(uploaded_file)
input_text = text_input or file_text or ""

# Input Statistics
if input_text.strip():
    words = len(input_text.split())
    reading_score = textstat.flesch_reading_ease(input_text)
    reading_time = round(words / 200, 2)
    complex_words = textstat.difficult_words(input_text)

    st.markdown("### 📊 Text Statistics")
    st.markdown(f"- 🧮 *Words*: {words}")
    st.markdown(f"- 📘 *Readability (Flesch Score)*: {round(reading_score, 2)}")
    st.markdown(f"- ⏱ *Estimated Reading Time*: {reading_time} min")
    st.markdown(f"- 🤯 *Difficult Words*: {complex_words}")

# Advanced Setting Toolbar
with st.expander("🔧 Advanced Settings", expanded=False):
    st.markdown("<div class='advanced-box'>", unsafe_allow_html=True)
    max_length = st.slider("📏 Max summary length", 30, 512, 150)
    min_length = st.slider("🧱 Min summary length", 10, 100, 40)
    num_beams = st.slider("🔭 Beam search (higher = better, slower)", 1, 10, 4)
    st.markdown("</div>", unsafe_allow_html=True)

# Summary Generation
if st.button("✨ Generate Summary"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=3
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary_words = len(summary.split())
            compression = round(100 * (summary_words / max(1, words)), 2)

        st.markdown('<div class="section-title">✅ AI-Generated Summary:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="compact-summary">{summary}</div>', unsafe_allow_html=True)
        st.code(summary, language='text')

        b64 = base64.b64encode(summary.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="summary.txt">📥 Download Summary</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.text_input("📋 Copy Summary:", summary)

        st.markdown("### 📉 Summary Stats")
        st.markdown(f"- 📝 *Original Words*: {words}")
        st.markdown(f"- ✂ *Summary Words*: {summary_words}")
        st.markdown(f"- 📉 *Compression Rate*: {compression}%")

        fig1, ax1 = plt.subplots(figsize=(2, 2))
        ax1.pie([summary_words, words - summary_words], labels=['Summary', 'Reduced'],
                autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF7043'])
        ax1.axis('equal')
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(["Original", "Summary"], [words, summary_words], color=['#2196F3', '#8BC34A'])
        ax2.set_ylabel("Word Count")
        ax2.set_title("Word Count Comparison")
        st.pyplot(fig2)

        sentiment = sentiment_pipeline(summary)[0]
        st.markdown("### 🎭 Summary Sentiment")
        st.markdown(f"- *Sentiment*: {sentiment['label']} with score {round(sentiment['score'], 2)}")
    else:
        st.warning("⚠ Please input text or upload a file.")

# Wikipedia Integration
with st.expander("🌐 Wikipedia Summarizer"):
    topic = st.text_input("Enter a topic (e.g., Artificial Intelligence):")
    if st.button("📚 Summarize Wikipedia Article"):
        try:
            article = wikipedia.summary(topic, sentences=10)
            st.markdown(f"### ✅ Summary of *{topic}* from Wikipedia")
            st.write(article)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# News API Integration
def fetch_news(api_key, topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&language=en&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json()["articles"][:3]
    return articles

with st.expander("📰 News Summarizer"):
    topic_news = st.text_input("Search latest news on:")
    news_api_key = st.text_input("🔑 Enter your NewsAPI Key", type="password")
    if st.button("📩 Fetch and Summarize News"):
        try:
            news_list = fetch_news(news_api_key, topic_news)
            for i, article in enumerate(news_list, 1):
                st.markdown(f"### {i}. {article['title']}")
                st.write(article["description"])
                st.markdown(f"[Read more]({article['url']})")
        except Exception as e:
            st.error(f"❌ Could not fetch news: {e}")