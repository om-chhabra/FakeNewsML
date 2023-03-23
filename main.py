
#Importing libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading the Dataset
data = pd.read_csv('fakenews.csv')
data.head()

#Data Preprocessing
data.shape
data.isnull().sum()
df1 = data.fillna('')
df1['content'] = df1['author'] + ' ' + df1['title']

# Stemming
stemmer = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content) #1
    stemmed_content = stemmed_content.lower() #2
    stemmed_content = stemmed_content.split() #3
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')] #4
    stemmed_content = ' '.join(stemmed_content) #5
    return stemmed_content #6
df1['content'] = df1['content'].apply(stemming)
df1['content'].head()
X = df1.content.values
y = df1.label.values
X = TfidfVectorizer().fit_transform(X)
print(X)

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)

# Training the Model
model = LogisticRegression()
model.fit(X_train, y_train)
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, y_train)
print(training_accuracy)
X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, y_test)
print(testing_accuracy)

#Building a system
X_sample = X_test[0]
prediction = model.predict(X_sample)
if prediction == 0:
    print('The NEWS is Real!')
else:
    print('The NEWS is Fake!')prediction = model.predict(X_sample)
