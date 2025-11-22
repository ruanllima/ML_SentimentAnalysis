import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

nltk.download("stopwords")
# execute 'python -m spacy download en_core_web_sm' in cmd to download the package
nlp = spacy.load("en_core_web_sm")


# adding dataset
data = pd.read_excel("./data/twiter_sentiment.xlsx")
df = pd.DataFrame(data)

# CLEAN DATAS

# remove unused columns
df = df.drop(columns=["none", "none.1"])
# check null rows
print(df.isna().sum())
# remove rows with NaN values
df = df[df["Text"].notna()]

# data balancing chart
qtdSentiments = []
labels = df['Sentiment'].unique()
for i in labels:
    qtd = df['Sentiment'].value_counts()[i]
    qtd = int(qtd)
    qtdSentiments.append(qtd)
print(qtdSentiments)

# ==== Remove - ''' - to show pie chart ====
'''fig, ax = plt.subplots()
ax.pie(qtdSentiments, labels=labels, autopct='%1.1f%%')
plt.show()''' 

# DEFINE TRAIN AND TEST DATAS
X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Sentiment"], random_state=42)

# Check X_train and y_train size's
print(f"X_train's size: {len(X_train)}")
print(f"y_train's size: {len(y_train)}")

# Remove numbers, initial characters, and convert to lowercase.
X_train = [re.sub(r'[^a-zA-Z\s]', '', text).lower() for text in X_train]
X_test = [re.sub(r'[^a-zA-Z\s]', '', text).lower() for text in X_test]

# Tokenization
X_train = [phrase.split() for phrase in X_train]
X_test = [phrase.split() for phrase in X_test]

# Remove Stop Words
stop_words = set(stopwords.words('english'))
X_train = [[word for word in phrase if word not in stop_words] for phrase in X_train]
X_test = [[word for word in phrase if word not in stop_words] for phrase in X_test]

# Lemmatization
X_train = [' '.join([token.lemma_ for token in nlp(' '.join(phrase))]) for phrase in X_train]
X_test = [' '.join([token.lemma_ for token in nlp(' '.join(phrase))]) for phrase in X_test]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Data balancing by combining SMOTE and ENN.
smote_enn = SMOTEENN() 
X_train, y_train = smote_enn.fit_resample(X_train.toarray(), y_train)


# ALGORITHM IMPLEMENTATION (SVM) 
algorithm = SVC()
# train
algorithm.fit(X_train, y_train)
# predict
# y_pred = algorithm.predict(X_test.toarray())
# metric
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy is: {accuracy}")


# ========================== MODEL =============================
joblib.dump(algorithm   , "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")