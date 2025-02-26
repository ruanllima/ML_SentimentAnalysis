# Sentiment Analysis with Multiclasses Classification Algorithms
<br>

## Introduction

<p>In the context of Machine Learning, the classification task can be approached in several ways, each with its own advantages and disadvantages. In this project, we will implement a variety of multi-class classification algorithms to identify the most effective model for analyzing sentiment in texts from social networks.</p>
<p>By the end of this project, we hope to not only identify the best classification model, but also gain a deeper understanding of how different text vectorization techniques and machine learning algorithms interact and influence the accuracy of sentiment analysis.</p>

## Dataset
<p>The dataset consists of data lines obtained from social networks. They contain information such as the main content, the described sentiment, and the obtained text. Four types of sentiment are characterized in it: Positive, Negative, Neutral, and Irrelevant. The dataset contains all the information in English. Below is an example of the lines in this dataset.</p>
<br>
<p align="center">
  <img width="600" src="/media/img1.png">
</p>

## Data Analysis
<p>The dataset initially needs some treatment. More specifically, empty rows and elements with a type other than 𝑠𝑡𝑟. Using the 𝑝𝑎𝑛𝑑𝑎𝑠 library, 858 rows with null elements were identified   and removed. In addition, 17 rows with content whose type was different from 𝑠𝑡𝑟 were removed.</p>
<p>Using the 𝑚𝑎𝑡𝑝𝑙𝑜𝑡𝑙𝑖𝑏 library we can observe the balance of our data:</p>
<p align="center">
  <img height="380" src="/media/img2.png">
</p>

## Data Pre-processing
<p>Since we are working with texts, this step is crucial. In it, we will remove any and all unnecessary information for our project. We will remove accents, punctuation, special characters and standardize our texts to lowercase. In addition, we will perform some processes such as tokenization, removal of stop words, stemming and lemmatization. Only after having done all this will we be able to continue with the project.</p>
<p></p>
<p>The <strong><em>re</em></strong> and <strong><em>ntlk</em></strong> libraries were used</p>

## Methodology
<p>Algorithms such as K-Nearest Neighbors (KNN), Decision Tree, Support Vector Machine (SVM), Naive Bayes, and Random Forest were used. All the algorithms used belong to the 𝑠𝑘𝑙𝑒𝑎𝑟𝑛 library. Additionally, this project explored two popular text vectorization techniques: TF-IDF (Term Frequency-Inverse Document Frequency) and Bag of Words (BoW). The combination of these techniques with different classification algorithms allowed for a comprehensive evaluation of their effectiveness in the task of sentiment analysis. With a balanced and large dataset, 𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦 was used as the main performance metric for the models. Libraries such as 𝑛𝑢𝑚𝑝𝑦 and 𝑚𝑎𝑡𝑝𝑙𝑜𝑡𝑙𝑖𝑏 were used for visualizing the results.
With this, the algorithms were implemented and the cross-analysis with the two types of vectorization was performed. See an example below:</p>
<p align="center">
  <img src="/media/img3.png">
</p>

## Results
<p>After defining, training and predicting each model, we obtain the following results:</p>
<p align="center">
  <img height="350" src="/media/img4.png">
</p>
<p>Below we can visualize these results graphically:</p>
<p align="center">
  <img height="370" src="/media/img5.png">
</p>
