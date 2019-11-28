import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
path='/media/ayman/8497-FE2D/x.json'
data_set = pd.read_csv('x.json',names=['text','label'])



X_train, X_test, y_train, y_test = train_test_split(data_set['text'],data_set['label'],random_state=0)
print("\ntrain data\n",X_train)
print('---------------------------------------\n test data\n',X_test)
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)

clfrNB = MultinomialNB(alpha = 0.1)
clfrNB.fit(X_train_vectorized, y_train)
preds = clfrNB.predict(vect.transform(X_test))
score = roc_auc_score(y_test, preds)
print('accuracy \n:',score)
value=None
for index,valu in enumerate(X_test):
    if(preds[index]==1):
        value='software';
    else:
        value="not software"
    print(valu,'--->',value)
