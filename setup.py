import nltk
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')