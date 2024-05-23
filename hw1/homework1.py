#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install scikit-learn


# In[360]:


import json
from collections import defaultdict
import sklearn
from sklearn import linear_model
import numpy
import random
import gzip
import dateutil.parser
import math


# In[361]:


answers = {}


# In[362]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[363]:


### Question 1


# In[364]:


f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[365]:


# def feature(datum):
    # ...


# In[366]:


X = [len(d['review_text']) for d in dataset]
max_Len = max(X)
X = [[1, d/max_Len] for d in X]

Y = [d['rating'] for d in dataset]


# In[367]:


theta,residuals,rank,s = numpy.linalg.lstsq(X, Y)


# In[368]:


answers['Q1'] = [theta[0], theta[1], (residuals/len(dataset))[0]]
print(answers['Q1'])


# In[369]:


assertFloatList(answers['Q1'], 3)


# In[370]:


### Question 2


# In[371]:


for d in dataset:
    t = dateutil.parser.parse(d['date_added'])
    d['parsed_date'] = t


# In[372]:


# 1 len 000000 00000000000 represent: mon jan... 
def feature(datum):
    x = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    date = datum['parsed_date']
    if(date.weekday()!=0):
        x[date.weekday()+1]=1
    if(date.month!=1):
        x[date.month+6]=1
    x[1]= datum['text_len']
    return x


# In[373]:


max_Len = 0
for d in dataset:
    d['text_len']= len(d['review_text'])
    if d['text_len']>max_Len:
        max_Len = d['text_len']
for d in dataset:
    d['text_len']= d['text_len']/max_Len
X = [feature(d) for d in dataset]
Y = [d['rating'] for d in dataset]


# In[374]:


answers['Q2'] = [X[0], X[1]]


# In[375]:


assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)


# In[376]:


### Question 3


# In[377]:


def feature3(datum):
    x=[1,0,0,0]
    x[1] = datum['text_len']
    x[2] = datum['parsed_date'].weekday()
    x[3] = datum['parsed_date'].month
    return x


# In[378]:


X3 = [feature3(d) for d in dataset]
Y3 = [d['rating'] for d in dataset]


# In[379]:


theta31,residuals31,rank31,s31 = numpy.linalg.lstsq(X, Y)

theta32,residuals32,rank32,s32 = numpy.linalg.lstsq(X3, Y3)


# In[380]:


answers['Q3'] = [(residuals31/len(dataset))[0], (residuals32/len(dataset))[0]]
print(answers['Q3'])


# In[381]:


assertFloatList(answers['Q3'], 2)


# In[382]:


### Question 4


# In[383]:


from sklearn.metrics import mean_squared_error


# In[384]:


random.seed(0)
random.shuffle(dataset)


# In[385]:


X2 = [feature(d) for d in dataset]
X3 = [feature3(d) for d in dataset]
Y = [d['rating'] for d in dataset]


# In[386]:


train2, test2 = X2[:len(X2)//2], X2[len(X2)//2:]
train3, test3 = X3[:len(X3)//2], X3[len(X3)//2:]
trainY, testY = Y[:len(Y)//2], Y[len(Y)//2:]


# In[387]:


theta2,residuals2,rank2,s2 = numpy.linalg.lstsq(train2, trainY)

theta3,residuals3,rank3,s3 = numpy.linalg.lstsq(train3, trainY)


# In[388]:


test2y = numpy.dot(theta2,numpy.array(test2).T)
test3y = numpy.dot(theta3,numpy.array(test3).T)


# In[389]:


answers['Q4'] = [mean_squared_error(test2y, testY), mean_squared_error(test3y, testY)]
print(answers['Q4'])


# In[390]:


assertFloatList(answers['Q4'], 2)


# In[391]:


### Question 5


# In[395]:


f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[396]:


print(dataset[0])


# In[397]:


X = [[1, len(d['review/text'])] for d in dataset]
y = [int(d['review/overall'] >= 4) for d in dataset]


# In[398]:


mod = sklearn.linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)


# In[399]:


predictions = mod.predict(X) # Binary vector of predictions
correct = predictions == y # Binary vector indicating which predictions were correct
sum(correct) / len(correct)


# In[400]:


TP = sum([(p and l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])


# In[401]:


TPR = TP / (TP + FN)
TNR = TN / (TN + FP)


# In[402]:


BER = 1 - 1/2 * (TPR + TNR)
BER


# In[403]:


answers['Q5'] = [TP, TN, FP, FN, BER]


# In[404]:


assertFloatList(answers['Q5'], 5)


# In[ ]:


### Question 6


# In[408]:


scores = mod.decision_function(X)


# In[411]:


scoreslabels = list(zip(scores, y))
scoreslabels.sort(reverse=True)
sortedlabels = [x[1] for x in scoreslabels]


# In[412]:


retrieved = sum(predictions)
relevant = sum(y)
intersection = sum([y and p for y,p in zip(y,predictions)])


# In[413]:


precs = []


# In[414]:


for k in [1,100,1000,10000]:
    precs.append(sum(sortedlabels[:k])/k)


# In[416]:


answers['Q6'] = precs


# In[ ]:


assertFloatList(answers['Q6'], 4)


# In[ ]:


### Question 7


# In[ ]:


dataset[:5]


# In[420]:


X7 = [[1, len(d['review/text']), int(d['review/appearance'] >= 4), int(d['review/palate'] >= 4), int(d['review/taste'] >= 4), int(d['review/aroma'] >= 4)] for d in dataset]
y7 = [int(d['review/overall'] >= 4) for d in dataset]


# In[421]:


mod = sklearn.linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X7,y7)


# In[422]:


predictions7 = mod.predict(X7) # Binary vector of predictions
correct7 = predictions7 == y7 # Binary vector indicating which predictions were correct
sum(correct7) / len(correct7)


# In[423]:


TP7 = sum([(p and l) for (p,l) in zip(predictions7, y7)])
TN7 = sum([(not p and not l) for (p,l) in zip(predictions7, y7)])
FP7 = sum([(p and not l) for (p,l) in zip(predictions7, y7)])
FN7 = sum([(not p and l) for (p,l) in zip(predictions7, y7)])


# In[424]:


TPR7 = TP7 / (TP7 + FN7)
TNR7 = TN7 / (TN7 + FP7)


# In[426]:


BER7 = 1 - 1/2 * (TPR7 + TNR7)
BER7


# In[427]:


its_test_BER = 1000


# In[428]:


answers['Q7'] = ["Considering the addition of beer 'review/appearance', 'review/palate', 'review/taste', and 'review/aroma' features, the balanced error rate decreases from 0.4683 to 0.1835, indicating that the PPR as well as the FNR decrease after the addition of features, suggesting that the prediction accuracy improves", BER7]


# In[ ]:


f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




