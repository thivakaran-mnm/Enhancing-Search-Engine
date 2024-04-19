import streamlit as st
import sqlite3
import pandas as pd
import zipfile
import io
import numpy as np
import string
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Function to decode binary data
def decode_method(binary_data):
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            subtitle_content = zip_file.read(zip_file.namelist()[0])
    return subtitle_content.decode('latin-1')

# Function to chunk document
def chunk_document(text, chunk_size=500, overlap=100):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(' '.join(words[i:i+chunk_size]))
    return chunks

# Function for text cleaning
def cleantext(texts):
    cleaned_texts = []
    stop_words = set(stopwords.words('english'))
    for text in texts:
        text = text.lower()
        tokens = word_tokenize(text)
        cleaned_token = []
        lemmatizer = WordNetLemmatizer()
        for word in tokens:
            if word.isalnum() and not word.isdigit() and word not in stop_words:
                word = word.strip(string.punctuation)
                word = lemmatizer.lemmatize(word)
                word = word.strip()
                cleaned_token.append(word)
            cleaned_text = " ".join(cleaned_token)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

# Function to retrieve relevant documents
def retrieve_documents(user_query, data, model, embeddings_dict):
    X = data['clean_text']
    query_embedding = model.encode([user_query])
    similarity_scores = {}
    for name, embedding in embeddings_dict.items():
        similarity_scores[name] = cosine_similarity(query_embedding, embedding.reshape(1, -1))[0][0]
    sorted_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_documents = sorted_documents[:10]  # Top 10 relevant documents
    return top_documents

# Streamlit App
def main():
    # Connect to SQLite database
    conn = sqlite3.connect('E:\INNOMATICS\subtitles_db.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = cursor.fetchall()
    
    # Read subtitles table into DataFrame
    df = pd.read_sql_query("""SELECT * FROM subtitles""", conn)
    
    # Decompress binary data
    df['file_content'] = df['content'].apply(decode_method)
    
    # Sample 30% of the data
    data = df.sample(frac=0.3, random_state=42)
    
    # Clean text data
    data['clean_text'] = cleantext(data['file_content'].values)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['clean_text'])
    
    # Load or create and save SentenceTransformer model
    model_path = "bert_model.pickle"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Load embeddings or create and save
    embeddings_dict_path = "embeddings_dict.pickle"
    try:
        with open(embeddings_dict_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
    except FileNotFoundError:
        embeddings_dict = {}
        for index, sentence in data['clean_text'].items():
            embedding = model.encode([sentence])
            embeddings_dict[data.iloc[index]['name']] = embedding.reshape(1, -1)
        with open(embeddings_dict_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    
    # User Input
    st.title("Subtitle Search Engine")
    user_query = st.text_input("Enter your search query:")
    
    # Process Query and Display Results
    if st.button("Search"):
        top_documents = retrieve_documents(user_query, data, model, embeddings_dict)
        st.write("Top 10 Relevant Documents:")
        for document, score in top_documents:
            st.write(f"Document: {document}, Similarity Score: {score}")

if __name__ == "__main__":
    main()
