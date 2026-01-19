import re
import nltk
import spacy

from nltk.corpus import stopwords

nltk.download('stopwords')

nlp = spacy.load("ru_core_news_md")

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\wа-яё]+", " ", text)
    doc = nlp(text)

    return [token.lemma_ for token in doc]

def preprocessing_text(text):
    text = tokenize(text)

    stop_words = set(stopwords.words('russian'))

    filtered_tokens = [word for word in text if word not in stop_words and word.isalpha()]

    return filtered_tokens