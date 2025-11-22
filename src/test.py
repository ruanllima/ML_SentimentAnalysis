import joblib
import re
import nltk
from nltk.corpus import stopwords
import spacy
import pandas
from googletrans import Translator

translator = Translator()
# Carrega os arquivos da mesma pasta
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
nltk.download("stopwords")
# execute 'python -m spacy download en_core_web_sm' in cmd to download the package
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join([token.lemma_ for token in nlp(' '.join(words))])
user_input = ''
while user_input != "exit":
    user_input = str(input("Digite um texto: "))
    user_input = translator.translate(user_input, src='pt', dest='en').text
    processed = preprocess(user_input)
    X = vectorizer.transform([processed])
    prediction = model.predict(X.toarray())[0]
    print(prediction)

