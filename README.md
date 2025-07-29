# Rag to Riches: Sentence Embedding API

## Overview

**Rag to Riches** is a Flask-based RESTful API that leverages the `sentence-transformers` library to convert input sentences into dense vector embeddings. These embeddings capture the semantic meaning of the text, enabling applications like semantic search, clustering, and similarity analysis.

## Core Components

### 1. **Sentence Embedding Model**

The API utilizes the `all-MiniLM-L6-v2` model from the `sentence-transformers` library. This model is designed to efficiently generate high-quality sentence embeddings, balancing performance and computational efficiency.

### 2. **Flask API**

The backend is built using Flask, a lightweight Python web framework. It provides a simple interface to expose the embedding functionality over HTTP.

### 3. **Deployment**

The application is deployed using Gunicorn, a Python WSGI HTTP Server, ensuring robust performance in production environments.

## Features

- **Sentence Embedding**: Convert input sentences into fixed-size vector representations.
- **Similarity Analysis**: Compute cosine similarity between sentence embeddings to assess semantic similarity.
- **Batch Processing**: Handle multiple sentences in a single request for efficient processing.

## Use Cases

- **Semantic Search**: Retrieve documents or sentences semantically similar to a query.
- **Clustering**: Group similar sentences together for topic modeling or categorization.
- **Recommendation Systems**: Suggest content based on semantic similarity to user preferences.

## Example Usage

### Request

```bash
curl -X POST "https://rag-to-riches-1.onrender.com/embed" -H "Content-Type: application/json" -d '{"sentence": "Your input sentence here."}'
