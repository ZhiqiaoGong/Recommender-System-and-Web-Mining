#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model
import random
import statistics
import sklearn


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


answers = {}


# In[4]:


z = gzip.open("train.json.gz")


# In[5]:


dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)


# In[6]:


z.close()


# In[7]:


### Question 1


# In[8]:


dataset[1]


# In[9]:


def MSE(y, ypred):
    return sum([(p - l)**2 for (p,l) in zip(y, ypred)])/len(y)


# In[10]:


def MAE(y, ypred):
    return sum([abs(p - l) for (p,l) in zip(y, ypred)])/len(y)


# In[11]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])
reviewsPerItem['g88735741']


# In[12]:


# def feat1(d):
    


# In[13]:


X = [[1, d['hours']] for d in dataset]
y = [len(d['text']) for d in dataset]


# In[14]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[15]:


_,p = mod.coef_


# In[16]:


answers['Q1'] = [p, MSE(y, predictions)]


# In[17]:


assertFloatList(answers['Q1'], 2)


# In[18]:


### Question 2


# In[19]:


mid = sum([d['hours'] for d in dataset])/len(X)


# In[20]:


def feat2(d):
    x = [1, 0, 0, 0, 0]
    x[1] = d['hours']
    x[2] = d['hours_transformed']
    x[3] = math.sqrt(d['hours'])
    x[4] = d['hours'] > mid
    return x


# In[21]:


X = [feat2(d) for d in dataset]


# In[22]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[23]:


answers['Q2'] = MSE(y,predictions)


# In[24]:


assertFloat(answers['Q2'])


# In[25]:


### Question 3


# In[26]:


def feat3(d):
    x = [1, 0, 0, 0, 0, 0]
    d = d['hours']
    x[1] = d > 1
    x[2] = d > 5
    x[3] = d > 10
    x[4] = d > 100
    x[5] = d > 1000
    return x


# In[27]:


X = [feat3(d) for d in dataset]


# In[28]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[29]:


answers['Q3'] = MSE(y, predictions)


# In[30]:


assertFloat(answers['Q3'])


# In[31]:


### Question 4


# In[32]:


def feat4(d):
    x = [1, 0]
    x[1] = len(d['text'])
    return x


# In[33]:


X = [feat4(d) for d in dataset]
y = [d['hours'] for d in dataset]


# In[34]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[35]:


mse = MSE(y, predictions)
mae = MAE(y, predictions)


# In[36]:


answers['Q4'] = [mse, mae, "Mae is better. Because there are many extremely large or small values in this dataset, mse is more sensitive to these values and will lead to larger results. Whereas mae is not so sensitive and can gain a more reasonable result."]
answers['Q4']


# In[37]:


assertFloatList(answers['Q4'][:2], 2)


# In[38]:


### Question 5


# In[39]:


y_trans = [d['hours_transformed'] for d in dataset]


# In[40]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[41]:


mse_trans = MSE(y_trans, predictions_trans)


# In[42]:


predictions_untrans = [math.pow(2, p)-1 for p in predictions_trans]


# In[43]:


mse_untrans = MSE(y, predictions_untrans)


# In[44]:


answers['Q5'] = [mse_trans, mse_untrans]
answers['Q5']


# In[45]:


assertFloatList(answers['Q5'], 2)


# In[46]:


### Question 6


# In[47]:


def feat6(d):
    x = []
    for i in range(100):
        x.append(0)
    h = d['hours']
    h = math.floor(h)
    x[0] = 1
    if h < 99:
        x[h] = 1
    else:
        x[99] = 1
    return x


# In[48]:


X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]


# In[49]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[50]:


models = {}
mse = {}
bestC = None

for c in [1, 10, 100, 1000, 10000]:
    mod = linear_model.Ridge(alpha = c)
    mod.fit(Xtrain,ytrain)
    predictions = mod.predict(Xvalid)
    msev = MSE(yvalid, predictions)
    models[c] = mod
    mse[c] = msev

bestmse = mse.get(1)
for c in [1, 10, 100, 1000, 10000]:
    if mse.get(c) < bestmse:
        bestmse = mse.get(c)
        bestC = c


# In[51]:


mod = models.get(bestC)


# In[52]:


predictions_test = mod.predict(Xtest)


# In[53]:


mse_valid = bestmse


# In[54]:


mse_test = MSE(ytest, predictions)


# In[55]:


answers['Q6'] = [bestC, mse_valid, mse_test]
answers['Q6']


# In[56]:


assertFloatList(answers['Q6'], 3)


# In[57]:


### Question 7


# In[58]:


times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)


# In[59]:


notPlayed = [t<1 for t in times]
nNotPlayed = sum(notPlayed)


# In[60]:


answers['Q7'] = [median, nNotPlayed]
answers['Q7']


# In[61]:


assertFloatList(answers['Q7'], 2)


# In[62]:


### Question 8


# In[63]:


def feat8(d):
    return [1, len(d['text'])]


# In[64]:


X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]


# In[65]:


mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions


# In[66]:


def rates(pred, y):
    TP = sum([(p and l) for (p,l) in zip(pred, y)])
    TN = sum([(not p and not l) for (p,l) in zip(pred, y)])
    FP = sum([(p and not l) for (p,l) in zip(pred, y)])
    FN = sum([(not p and l) for (p,l) in zip(pred, y)])
    return TP, TN, FP, FN


# In[67]:


TP, TN, FP, FN = rates(predictions, y)


# In[68]:


TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)


# In[69]:


answers['Q8'] = [TP, TN, FP, FN, BER]
answers['Q8']


# In[70]:


assertFloatList(answers['Q8'], 5)


# In[71]:


### Question 9


# In[72]:


scores = mod.decision_function(X)
scoreslabels = list(zip(scores, y))
scoreslabels.sort(reverse=True)
sortedlabels = [x[1] for x in scoreslabels]


# In[73]:


# precision = 
# recall = 


# In[74]:


precs = []
recs = []

for i in [5, 10, 100, 1000]:
    si = scoreslabels[i][0]
    while scoreslabels[i][0] == si:
        i+=1
    precs.append(sum(sortedlabels[:i])/i)
precs


# In[75]:


answers['Q9'] = precs


# In[76]:


assertFloatList(answers['Q9'], 4)


# In[77]:


### Question 10


# In[78]:


y_trans = [d['hours_transformed'] for d in dataset]


# In[79]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[80]:


myth = 3.694
predictions_thresh = [p > myth for p in predictions_trans]
y_thresh = [y > myth for y in y_trans]

TP, TN, FP, FN = rates(predictions_thresh, y)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)
BER


# In[81]:


answers['Q10'] = [myth, BER]
answers['Q10']


# In[82]:


assertFloatList(answers['Q10'], 2)


# In[83]:


### Question 11


# In[84]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[85]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataTrain:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])


# In[86]:


userMedian = defaultdict(set)
itemMedian = defaultdict(set)

for ru in reviewsPerUser:
    tu = [u['hours'] for u in reviewsPerUser[ru]]
    userMedian[ru] = statistics.median(tu)

for ri in reviewsPerItem:
    ti = [i['hours'] for i in reviewsPerItem[ri]]
    itemMedian[ri] = statistics.median(ti)


# In[87]:


answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]
answers['Q11']


# In[88]:


assertFloatList(answers['Q11'], 2)


# In[89]:


### Question 12


# In[90]:


hoursMed = statistics.median([d['hours'] for d in dataTrain])
hoursMed


# In[91]:


def f12(u,i):
    if itemMedian[i] is not None:
        if itemMedian[i] > hoursMed:
                return 1
        else:
            return 0
    elif userMedian[u] is not None:
        if userMedian[u] > hoursMed:
            return 1
        else:
            return 0
    else:
        return 0


# In[92]:


preds = [f12(d['userID'], d['gameID']) for d in dataTest]


# In[93]:


y = [1 if d['hours'] > hoursMed else 0 for d in dataTest]


# In[94]:


accuracy = [preds[i]==y[i] for i in range(len(preds))]
accuracy = sum(accuracy)/len(accuracy)
accuracy


# In[95]:


answers['Q12'] = accuracy


# In[96]:


assertFloat(answers['Q12'])


# In[97]:


### Question 13


# In[98]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)


# In[99]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[100]:


def mostSimilar(i, func, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = func(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[101]:


ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10)


# In[102]:


answers['Q13'] = [ms[0][0], ms[-1][0]]
answers['Q13']


# In[103]:


assertFloatList(answers['Q13'], 2)


# In[104]:


### Question 14


# In[105]:


def mostSimilar14(i, func, N):
    similarities = []
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = func(i, i2)
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[106]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = 1 if d['hours'] > hoursMed else -1
    ratingDict[(u,i)] = lab


# In[107]:


def Cosine(i1, i2):
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += ratingDict[(u,i1)]*ratingDict[(u,i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u,i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[108]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[109]:


answers['Q14'] = [ms[0][0], ms[-1][0]]
answers['Q14']


# In[110]:


assertFloatList(answers['Q14'], 2)


# In[111]:


### Question 15


# In[112]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed']
    ratingDict[(u,i)] = lab


# In[113]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[114]:


answers['Q15'] = [ms[0][0], ms[-1][0]]
answers['Q15']


# In[115]:


assertFloatList(answers['Q15'], 2)


# In[116]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




