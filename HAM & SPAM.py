#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv("/Users/sayedrizwan/Downloads/spam.csv", encoding='ISO-8859-1')


# In[4]:


df.head(5)


# In[5]:


df.shape


# Data Cleaning

# In[6]:


df.info()


# In[7]:


#dropping the columns that have null values
df.drop(columns = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace = True)


# In[8]:


df.info()


# In[9]:


#renaming column names
df.rename(columns = {"v1" : "Target", "v2" : "Text"}, inplace = True)


# In[10]:


df.head(5)


# In[11]:


#Applying label encoder on Target Column
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df["Target"] = encoder.fit_transform(df["Target"])


# In[12]:


df.head(5)


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


#Dropping duplicates
df = df.drop_duplicates(keep = "first")


# In[16]:


df.duplicated().sum()


# EDA

# In[17]:


df["Target"].value_counts()


# In[18]:


import matplotlib.pyplot as plt
plt.pie(df["Target"].value_counts(), labels = ["HAM", "SPAM"], autopct = "%0.2f")
plt.show()


# In[19]:


import nltk


# In[20]:


#fetching number of characters in each message
df["num_characters"] = df["Text"].apply(len)


# In[21]:


df.head(5)


# In[22]:


#fetching number of words in each message


# In[23]:


df["num_words"] = df["Text"].apply(lambda x:len(nltk.word_tokenize(x)))


# In[24]:


df["num_sentences"] = df["Text"].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[25]:


df.head(5)


# In[26]:


#Analyzing all the newly created columns
df[["num_characters", "num_words", "num_sentences"]].describe()


# In[27]:


#For Ham meassges (Target = 0)
df[df["Target"] == 0][["num_characters", "num_words", "num_sentences"]].describe()


# In[28]:


#For Spam meassges (Target = 1)
df[df["Target"] == 1][["num_characters", "num_words", "num_sentences"]].describe()


# In[29]:


import seaborn as sns


# In[30]:


plt.figure(figsize = (12,8))
sns.histplot(df[df["Target"] == 0]["num_characters"])
sns.histplot(df[df["Target"] == 1]["num_characters"], color = "red")


# In[31]:


plt.figure(figsize = (12,8))
sns.histplot(df[df["Target"] == 0]["num_words"])
sns.histplot(df[df["Target"] == 1]["num_words"], color = "red")


# Data Preprocessing
# 
# 1. Lower Case
# 2. Tokenization
# 3. Removing special characters
# 4. Removing stop words and punctuations
# 5. Stemming

# In[32]:


from nltk.corpus import stopwords


# In[33]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[34]:


import string


# In[35]:


string.punctuation


# In[36]:


def transform_text(Text):
    Text = Text.lower()
    Text = nltk.word_tokenize(Text)
    
    y = []
    for i in Text:
        if i.isalnum():
            y.append(i)
            
    Text = y[:]
    y.clear()
    
    for i in Text:
        if i not in stopwords.words ("english") and i not in string.punctuation:
            y.append(i)
            
    Text = y[:]
    y.clear()
    
    for i in Text:
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[37]:


#Applying the above class on Text column
df["Transformed_Text"] = df["Text"].apply(transform_text)


# In[38]:


df.head(5)


# In[39]:


#Generating wordcloud to get the most important words
from wordcloud import WordCloud
wc = WordCloud(width = 600, height = 600, max_font_size = 60, background_color = "white")


# In[40]:


spam_wc = wc.generate(df[df["Target"] ==1]["Transformed_Text"].str.cat(sep = ""))


# In[41]:


plt.imshow(spam_wc)


# In[42]:


spam_corpus = []
for msg in df[df["Target"] == 1]["Transformed_Text"].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[43]:


len(spam_corpus)


# In[44]:


from collections import Counter
pd.DataFrame(Counter(spam_corpus).most_common(10))


# In[45]:


ham_corpus = []
for msg in df[df["Target"] == 0]["Transformed_Text"].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[46]:


len(ham_corpus)


# In[47]:


pd.DataFrame(Counter(ham_corpus).most_common(10))


# ****Model Building****

# In[48]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()


# In[49]:


X = cv.fit_transform(df["Transformed_Text"]).toarray()
X1 = tfidf.fit_transform(df["Transformed_Text"]).toarray()


# In[50]:


X.shape


# In[51]:


y = df["Target"].values
y1 = df["Target"].values


# In[52]:


y


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 42)


# In[54]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# In[55]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[56]:


gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

print(accuracy_score(y_test, y_pred_gnb))
print(confusion_matrix(y_test, y_pred_gnb))
print(precision_score(y_test, y_pred_gnb))


# In[57]:


mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

print(accuracy_score(y_test, y_pred_mnb))
print(confusion_matrix(y_test, y_pred_mnb))
print(precision_score(y_test, y_pred_mnb))


# In[58]:


bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

print(accuracy_score(y_test, y_pred_bnb))
print(confusion_matrix(y_test, y_pred_bnb))
print(precision_score(y_test, y_pred_bnb))


# In[59]:


gnb.fit(X1_train, y1_train)
y1_pred_gnb = gnb.predict(X1_test)

print(accuracy_score(y1_test, y1_pred_gnb))
print(confusion_matrix(y1_test, y1_pred_gnb))
print(precision_score(y1_test, y1_pred_gnb))


# In[60]:


mnb.fit(X1_train, y1_train)
y1_pred_mnb = mnb.predict(X1_test)

print(accuracy_score(y1_test, y1_pred_mnb))
print(confusion_matrix(y1_test, y1_pred_mnb))
print(precision_score(y1_test, y1_pred_mnb))


# In[61]:


bnb.fit(X1_train, y1_train)
y1_pred_bnb = bnb.predict(X1_test)

print(accuracy_score(y1_test, y1_pred_bnb))
print(confusion_matrix(y1_test, y1_pred_bnb))
print(precision_score(y1_test, y1_pred_bnb))


# In[62]:


import xgboost as xgb


# After observing all the predictions carefully we will be choosing Tfidf - Multinomial Naive Bayes. 
# After studing confusion matrix, we notice that it has 0 False Positive.

# In[63]:


#Now we will be importing all the important algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[64]:


#We will be creating objects of all the algorithms

lr = LogisticRegression(solver = "liblinear", penalty = "l1")
svc =SVC(kernel = "sigmoid", gamma = 0.1)
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
dtc = DecisionTreeClassifier(max_depth = 5)
knc = KNeighborsClassifier()
rfc = RandomForestClassifier(n_estimators = 50, random_state = 42)
abc = AdaBoostClassifier(n_estimators = 50, random_state = 42)
bc = BaggingClassifier(n_estimators = 50, random_state = 42)
etc = ExtraTreesClassifier(n_estimators = 50, random_state = 42)
gbc = GradientBoostingClassifier(n_estimators = 50, random_state = 42)
xgbc = XGBClassifier(n_estimators = 50, random_state = 42)


# In[65]:


#We will be creating dictionaries with keys at the algorithm names and values as its corresponding objects name

clfs = {
   "Logistic Regression" : lr,
   "SVC" : svc,
   "GaussianNB" : gnb,
   "MultinomialNB" : mnb,
   "BernoulliNB" : bnb,
   "Decision Tree Classifier" : dtc,
   "K-Neighbors Classifier" : knc,
   "Random Forest Classifier" : rfc,
   "Ada Boost Classifier" : abc,
   "Bagging Classifier" : bc,
   "Extra Trees Classifier" : etc,
   "Gradient Boosting Classifier" : gbc,
   "XGB Classifier" : xgbc
}


# In[66]:


#We will be building a class where we would be giving clf and X and y parameters

def train_classifier (clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    return accuracy, precision


# In[67]:


#Now we will be running a loop on the dictionary that we have created above

accuracy_scores = []
precision_scores = []
classifier_names = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    
    #print("For)
    #print("Accuracy", current_accuracy)
    #print("Precision", current_precision)
    
    classifier_names.append(name)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    


# In[68]:


performance_df = pd.DataFrame({
    "Algorithm" : classifier_names, 
    "Accuracy": accuracy_scores, 
    "Precision": precision_scores})


# In[69]:


# Sort the DataFrame by Accuracy
performance_df = performance_df.sort_values("Accuracy", ascending=False)


# In[70]:


performance_df


# In[71]:


#The pd.melt function in pandas is used to transform or reshape a DataFrame from a wide format to a long format.

performance_df1 = pd.melt(performance_df, id_vars = "Algorithm" )


# In[72]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# From above graph it can be noted that 
# 1. SVC
# 2. Extra Tree Classifier
# 3. Bernoulli NB
# 4. Random Forest Claasifier
# have good accuracy score as well as good precision score

# In[73]:


top_classifiers = performance_df[(performance_df["Accuracy"] > 0.95) & (performance_df["Precision"] > 0.95)]
top_classifiers


# After applying filter, it can be observed that 4 algorithms that we had selected by analyzing the graph have values gretaer than 0.95 for both Accuracy and Precision.

# Now we will be applying voting classifier to select the best performing model form the above 4 options.

# In[74]:


#Applying voting claassifier

svc =SVC(kernel = "sigmoid", gamma = 1.0, probability = True)
bnb = BernoulliNB()
rfc = RandomForestClassifier(n_estimators = 50, random_state = 42)
etc = ExtraTreesClassifier(n_estimators = 50, random_state = 42)

from sklearn.ensemble import VotingClassifier


# In[75]:


voting = VotingClassifier(estimators = [("svm", svc), ("bn", bnb), ("rc", rfc), ("ec", etc)], voting = "soft")


# In[76]:


voting.fit(X_train, y_train)


# In[77]:


y_pred_vote = voting.predict(X_test)

print(accuracy_score(y_test, y_pred_vote))
print(precision_score(y_test, y_pred_vote))


# The accuracy and precision scores are better than 3 classifiers out of 4. Extra Tree Classifier is still performing better than the voting classifier.
# 
# So we would be trying one another technique.

# In[78]:


from sklearn.ensemble import StackingClassifier


sc = StackingClassifier(estimators=[("svm", svc), ("bn", bnb), ("rc", rfc), ("ec", etc)],
                        final_estimator=RandomForestClassifier())


# In[79]:


sc.fit(X_train, y_train)

y_pred_sc = sc.predict(X_test)

print(accuracy_score(y_test, y_pred_sc))
print(precision_score(y_test, y_pred_sc))


# In[83]:


model = ExtraTreesClassifier()
model.fit(X1_train, y1_train)


# Even this model is not peforming better than Extra Tree Classifier

# So we would be using Extra Tree Classifier as it is the best performing model for us.
