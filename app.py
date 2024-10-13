from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import io
import base64

nltk.download('stopwords')

app = Flask(__name__)


# Fetch dataset and process
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize TF-IDF Vectorizer
stop_words = list(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)  # Limit to top 1000 words
X = vectorizer.fit_transform(documents)

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=100, random_state=42) # Reduce to 100 dimensions
X_reduced = svd.fit_transform(X)

def process_query(query):
    """ Transform user query into the reduced LSA space """
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)
    return query_reduced


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    query_reduced = process_query(query)
    similarities = cosine_similarity(query_reduced, X_reduced).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_documents = [documents[i] for i in top_indices]
    top_similarities = similarities[top_indices]
    return top_documents, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({
        'documents': documents,
        'similarities': similarities.tolist(),  # Convert ndarray to list
        'indices': indices.tolist()             # Convert ndarray to list
    })


def create_similarity_chart(similarities):
    """ Generate bar chart for similarity scores """
    fig, ax = plt.subplots()
    ax.barh(range(len(similarities)), similarities, color='blue')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Top 5 Documents')
    ax.set_title('Cosine Similarity of Top 5 Documents to Query')
    # Convert plot to PNG image and then to base64 string
    png_image = io.BytesIO()
    plt.savefig(png_image, format='png')
    png_image.seek(0)
    chart_data = base64.b64encode(png_image.getvalue()).decode('utf-8')
    return chart_data

if __name__ == '__main__':
    app.run(debug=True)
