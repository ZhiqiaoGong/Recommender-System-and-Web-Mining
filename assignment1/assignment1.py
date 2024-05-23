#!/usr/bin/env python
# coding: utf-8

# In[508]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import pandas
from sklearn.metrics import mean_squared_error


# In[509]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[510]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[511]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[512]:


answers = {}


# In[513]:


def UsersGames(data):
    usergames = defaultdict(set)
    gameusers = defaultdict(set)
    for u, g, d in data:
        usergames[u].add(g)
        gameusers[g].add(u)
    return usergames, gameusers


# In[514]:


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)


# In[515]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]


# In[516]:


##################################################
# Play prediction                                #
##################################################


# In[517]:


gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in hoursTrain:
  gameCount[game] += 1
  totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPlayed/2: break

def get_prediction(u,g):
    if g in return1:
        return 1
    else:
        return 0


# In[518]:


### Question 1


# In[519]:


# Evaluate baseline strategy
allGames = set([g for u, g, d in hoursTrain + hoursValid])

usergames,gameusers = UsersGames(hoursTrain + hoursValid)

augmentedValidationSet = []

valid1 = []
valid2 = []

for u,i,d in hoursValid:
    valid1.append((u, i, 1))
    valid2.append((u, random.choice(list(allGames - usergames[u])), 0))
augmentedValidationSet = valid1 + valid2


# In[520]:


predictions = [1 if get_prediction(u, g)==p else 0 for u,g,p in augmentedValidationSet]

accruacy = sum(predictions)/len(augmentedValidationSet)


# In[521]:


accruacy


# In[522]:


answers['Q1'] = accruacy


# In[523]:


assertFloat(answers['Q1'])


# In[524]:


### Question 2


# In[525]:


# Improved strategy
threshold = 2 * totalPlayed / 3

return12 = set()
count2 = 0
for ic, i in mostPopular:
  count2 += ic
  return12.add(i)
  if count2 > threshold: break

def get_prediction2(u,g):
    if g in return12:
        return 1
    else:
        return 0


# In[526]:


# Evaluate baseline strategy
predictions = [1 if get_prediction2(u, g)==p else 0 for u,g,p in augmentedValidationSet]

accruacy = sum(predictions)/len(augmentedValidationSet)


# In[527]:


accruacy


# In[528]:


answers['Q2'] = [accruacy, 100*2/3]


# In[529]:


assertFloatList(answers['Q2'], 2)


# In[530]:


### Question 3/4


# In[531]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[532]:


usergames_train,gameusers_train = UsersGames(hoursTrain)


# In[564]:


def predictJaccard(u,g):
    maxJaccrd = 0
    threshold34 = 0.03035
    predictionsJaccard = []
    gamesset = usergames_train.get(u)
    if gamesset is None:
        return 0
    for g1 in gamesset:
        if g==g1:
            continue
        j = Jaccard(gameusers_train.get(g1),gameusers_train.get(g))
        if j>maxJaccrd:
            maxJaccrd = j
    if maxJaccrd > threshold34:
        return 1
    else:
        return 0


# In[571]:


predictions = [predictJaccard(u, g)==p for u,g,p in augmentedValidationSet]

accruacy3 = sum(predictions)/len(augmentedValidationSet)


# In[572]:


accruacy3


# In[567]:


threshold = 2 * totalPlayed / 3

return13 = set()
count3 = 0
for ic, i in mostPopular:
  count3 += ic
  return13.add(i)
  if count3 > threshold: break

def get_prediction3(u,g):
    if g in return13:
        return 1
    else:
        return 0


# In[594]:


predictions = [1 if get_prediction3(u, g) or predictJaccard(u, g) else 0 for u,g,p in augmentedValidationSet]
accuracy4 = sum([p == l for p, l in zip([p for u,d,p in augmentedValidationSet], predictions)]) / len(augmentedValidationSet)
accuracy4


# In[595]:


answers['Q3'] = accruacy3
answers['Q4'] = accruacy4


# In[596]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[597]:


predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    baselinepre = get_prediction(u, g)
    jpre = predictJaccard(u, g)
    if baselinepre==p or jpre==p:
        pred = 1
    else:
        pred = 0
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[598]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[162]:


##################################################
# Hours played prediction                        #
##################################################


# In[163]:


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[599]:


### Question 6


# In[601]:


hoursPerUser = {}
hoursPerItem = {}
for u, g, d in hoursTrain:
    t = d.get('hours_transformed')
    if u not in hoursPerUser:
        hoursPerUser[u] = set()
    hoursPerUser[u].add(t)
    if g not in hoursPerItem:
        hoursPerItem[g] = set()
    hoursPerItem[g].add(t)


# In[602]:


betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0


# In[603]:


alpha = globalAverage # Could initialize anywhere, this is a guess
alpha


# In[604]:


lambda_ = 1
iter = 100
tolerance = 1e-5 
pmse = 0
converged = False


# In[605]:


usergameHours = {}
for u, g, d in hoursTrain:
    usergameHours[(u,g)] = d['hours_transformed']


# In[606]:


def iterate(lamb):
    alpha = globalAverage
    betaU = {}
    betaI = {}
    for u in hoursPerUser:
        betaU[u] = 0
    
    for g in hoursPerItem:
        betaI[g] = 0
    iter = 100
    tolerance = 1e-5 
    pmse = 0
    converged = False
    
    for iteration in range(iter):
        alpha = sum([ usergameHours[d] - betaU[d[0]] - betaI[d[1]] for d in usergameHours])/len(usergameHours)
    
        for u in usergames_train:
            betaU[u] = sum([ usergameHours[(u,g)] - alpha - betaI[g] for g in usergames_train[u]])/(lamb + len(usergames_train[u]))
    
        for g in gameusers_train:
            betaI[g] = sum([ usergameHours[(u,g)] - alpha - betaU[u] for u in gameusers_train[g]])/(lamb + len(gameusers_train[g]))
    
        predictions = [alpha + betaU[u] + betaI[g] for u,g,d in hoursValid]
        y = [d['hours_transformed'] for u,g,d in hoursValid]
    
        mse = mean_squared_error(predictions, y)
    
        if abs(pmse - mse) < tolerance:
            converged = True
            break
        else:
            pmse = mse
    return betaU, betaI, mse


# In[607]:


betaU, betaI, mse1 = iterate(1)


# In[608]:


answers['Q6'] = mse1


# In[609]:


assertFloat(answers['Q6'])


# In[610]:


### Question 7


# In[611]:


betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')


# In[612]:


answers['Q7'] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]


# In[613]:


answers['Q7']


# In[614]:


assertFloatList(answers['Q7'], 4)


# In[615]:


### Question 8


# In[616]:


# Better lambda...
_,_,bestmse = iterate(1)
for l in range(10):
    _,_,t = iterate(l)
    if t < bestmse:
        bestmse = t
        break


# In[617]:


answers['Q8'] = (l, bestmse)


# In[618]:


assertFloatList(answers['Q8'], 2)


# In[619]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    bu = betaU[u]
    bi = betaI[g]
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()


# In[620]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




