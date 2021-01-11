# Twitter-US-Airline-Sentiment-Analysis
In this repository I have utilised 6 different NLP Variations to predict the sentiments of the user as per the twitter reviews on airline. The dataset is 
Twitter US Airline Sentiment. The dataset has been imported from Kaggle with the following link:- 
https://www.kaggle.com/crowdflower/twitter-airline-sentiment/download
The text preprocessing involved removal of stopwords,punctuations and lemmatization taking care of POS Tags.I have used six methodologies for this classification.
<br>
**1) Without Any Vectorization**
<br>
Here I have just used a dictionary of most frequent 2500 words. So my training set includes a dictionary of top 2500 words after text preprocessing with true 
or false as the values of dictionary whether they occured in the sentence or not. Here I have used all 3 labels positive,negative and neutral and plotted a 
confusion matrix. The accuracy was observed around 78% accuracy with 86-87% of precison and recall.
<br>
**2) Using Machine Learning Algorithms like Naive Bayes, K Nearest Neighbors and Random Forests**
<br>
Here I used vectorization techniques such as Bag of Words, TF-IDF and word2Vec to use textual information and utilised the above machine learning algorithms
along with hyperparameter tuning for sentiment analysis. The Multinomial Naive Bayes Model acheived 89% accuracy and 0.95 AUC score while the KNN and Random Forest 
Models acheived accuracies of around 85-87% and AUC scores of 0.92.
<br>
**3) Using Deep Learning**
<br>
**Part A)** Here I used Artificial Neural Networks with only Dense Layers and two Dropout Layers. Initially I displayed the effect of regularization and dropout layer
on our prediction. After that I did hyperparamter tuning using **Keras Tuner** and acheived test and validation set accuracies of around 92-93% 
<br>
**Part B)** Here I used Embedding Layers and LSTMs alongside Dense and Dropout Layers. I further did hyperparameter tuning using Keras Tuner and acheived 98% accuracy
on test set and around 93% on validation set.
