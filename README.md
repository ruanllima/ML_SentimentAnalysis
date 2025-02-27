# Sentiment Analysis with Multiclass Classification Algorithms
<br>

## Technologies and dependencies
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable)


<p>The main language used in this project was <strong><i>Python</i></strong>, and the libraries used were</p>
<ul>
  <li>
  Pandas</li>
  <li>
  Scikit-learn</li>
  <li>
  Matplotlib</li>
  <li>
  NLTK</li>
</ul>

<br>

## Introduction

<p>In this project, various classification algorithms are implemented to determine the most effective model for sentiment analysis of social media texts. The goal is not only to identify the best algorithm but also to better understand how different text vectorization techniques impact model performance.
</p>

## Dataset
<p>The dataset consists of data lines obtained from social networks. They contain information such as the main content, the described sentiment, and the obtained text. Four types of sentiment are characterized in it: Positive, Negative, Neutral, and Irrelevant. The dataset contains all the information in English. Below is an example of the lines in this dataset.</p>
<br>
<p align="center">
  <img width="500" src="./media/img1.png">
</p>

## Data Analysis
<p>The dataset initially needs some treatment. More specifically, empty rows and elements with a type other than 𝑠𝑡𝑟. Using the <strong><i>pandas</i></strong> library, 858 rows with null elements were identified and removed. In addition, 17 rows with content whose type was different from 𝑠𝑡𝑟 were removed.</p>
<p>Using the <strong><i>matplotlib</i></strong> library we can observe the balance of our data:</p>
<p align="center">
  <img height="350" src="./media/img2.png">
</p>

## Data Pre-processing
<p>Since we are working with texts, this step is crucial. In it, we will remove any and all unnecessary information for our project. We will remove accents, punctuation, special characters and standardize our texts to lowercase. In addition, we will perform some processes such as tokenization, removal of stop words, stemming and lemmatization. Only after having done all this will we be able to continue with the project. For this, we used the <strong><i>re</i></strong> and <strong><i>ntlk</i></strong> libraries.</p>

## Methodology
<p>Algorithms such as <strong><i>K-Nearest Neighbors (KNN)</i></strong>, <strong><i>Decision Tree</i></strong>, <strong><i>Support Vector Machine (SVM)</i></strong>, <strong><i>Naive Bayes</i></strong>, and <strong><i>Random Forest</i></strong> were used. All the algorithms used belong to the <strong><i>sklearn</i></strong> library. Additionally, this project explored two popular text vectorization techniques: <strong><i>TF-IDF</i></strong> (Term Frequency-Inverse Document Frequency) and <strong><i>Bag of Words (BoW)</i></strong>. The combination of these techniques with different classification algorithms allowed for a comprehensive evaluation of their effectiveness in the task of sentiment analysis. With a balanced and large dataset, <strong><i>accuracy</i></strong> was used as the main performance metric for the models. Libraries such as <strong><i>numpy</i></strong> and <strong><i>matplotlib</i></strong> were used for visualizing the results. With this, the algorithms were implemented and the cross-analysis with the two types of vectorization was performed. See an example below:</p>
<p align="center">
  <img width="600"src="./media/img3.png">
</p>
<br>

## Results
<p>After defining, training and predicting each model, we obtain the following results:</p>
<p align="center">
  <img height="300" src="./media/img4.png">
</p>
<p>Below we can visualize these results graphically:</p>
<p align="center">
  <img height="340" src="./media/img5.png">
</p>
<br>
<p>The three models with the best accuracy were:</p>
<div align="center">
  <strong>
    <i>
    <h3>KNN + TFIDF</h3>
    <h3>KNN + TFIDF</h3>
    <h3>SVM + TFIDF</h3>
    </i>
  </strong>
</div>
<br>

### Perfomance of models with TF-IDF vectorization
<p align="center">
  <img height="350" src="./media/img6.png">
</p>
<p>The best one were <i><strong>KNN</strong>, <strong>SVM</strong></i> and <i><strong>Random Forest</strong></i>.</p>
<br>

### Perfomance of models with BoW vectorization
<p align="center">
  <img height="370" src="./media/img7.png">
</p>
<p>The best one were also <i><strong>KNN</strong>, <strong>SVM</strong></i> and <i><strong>Random Forest</strong></i>.</p>
<br>
<p>Below we can see the perfomance of each algorithm with each vectorization:</p>
<p align="center">
  <img height="350" src="./media/img8.png">
</p>
<p><strong><i>TF-IDF</strong></i> was superior in most models. Furthermore, the biggest difference between the two types can be observed in <strong><i>KNN</i></strong>, while the other algorithms the difference is minimal.</p>
<br>
<p>Overall, the average accuracy of all algorithms, only <strong><i>TF-IDF</i></strong>, only <strong><i>BoW</i></strong> was very similar</p>
<p align="center">
  <img height="330" src="./media/img9.png">
</p>
<br>
<h3>Average accuracy for each model:</h3>
<p align="center">
  <img height="360" src="./media/img10.png">
</p>
<br>
<h1>Conclusion</h1>
<p>The best performing models were <strong><i>KNN</i></strong>, <strong><i>SVM</i></strong>, and <strong><i>Random Forest</i></strong>, with <strong><i>Random Forest</i></strong> showing the highest stability and generalization ability. <strong><i>TF-IDF</i></strong> vectorization outperformed <strong><i>Bag of Words</i></strong> in most cases, leading to better sentiment analysis results.</p>

<br>
<h1>References</h1>
<p>
  Specionate, (2020). Twitter Sentiment Analysis.
  <a target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis">https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis</a>
  [Acesso em Julho de 2024].
</p>
<p>
  Genari J., (2022). Sarcastic headlines classification.
  <a target="_blank" rel="noopener noreferrer" href="https://nanogennari.medium.com/sarcastic-headlines-classification-9738b1541229">https://nanogennari.medium.com/sarcastic-headlines-classification-9738b1541229</a>
  [Acesso em Julho de 2024].
</p>
