#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from implicit import bpr


# In[8]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[9]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[10]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[ ]:


##################################################
# Play prediction                                #
##################################################


# In[11]:


answers = {}


# In[12]:


allHours = []
for l in readJSON("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/train.json.gz"):
    allHours.append(l)


# In[13]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g,r))
    hoursPerItem[g].append((u,r))


# In[24]:


allHours[0]


# In[43]:


def UsersGames(data):
    usergames = defaultdict(set)
    gameusers = defaultdict(set)
    for u, g, d in data:
        usergames[u].add(g)
        gameusers[g].add(u)
    return usergames, gameusers


# In[44]:


userSet = set()
gameSet = set()
playedSet = set()

for u,g,d in allHours:
    userSet.add(u)
    gameSet.add(g)
    playedSet.add((u,g))

lUserSet = list(userSet)
lGameSet = list(gameSet)

# notPlayedValid = set()
# for u,g,d in hoursValid:
#     g = random.choice(lGameSet)
#     while (u,g) in playedSet or (u,g) in notPlayedValid:
#         g = random.choice(lGameSet)
#     notPlayedValid.add((u,g))

# playedValid = set()
# for u,g,r in hoursValid:
#     playedValid.add((u,g))
allGames = set([g for u, g, d in hoursTrain + hoursValid])

usergames,gameusers = UsersGames(hoursTrain + hoursValid)

augmentedValidationSet = []

valid1 = []
valid2 = []

for u,i,d in hoursValid:
    valid1.append((u, i, 1))
    valid2.append((u, random.choice(list(allGames - usergames[u])), 0))
augmentedValidationSet = valid1 + valid2


# In[36]:


userIDs,gameIDs = {},{}

for u,i,d in allHours:
    u,i = d['userID'],d['gameID']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in gameIDs: gameIDs[i] = len(gameIDs)

nUsers,nGames = len(userIDs),len(gameIDs)


# In[19]:


Xui = scipy.sparse.lil_matrix((nUsers, nGames))
for u,g,d in allHours:
    Xui[userIDs[u],gameIDs[g]] = 1
    
Xui_csr = scipy.sparse.csr_matrix(Xui)


# In[231]:


model = bpr.BayesianPersonalizedRanking(factors = 70)


# In[232]:


model.fit(Xui_csr)


# In[233]:


recommended = model.recommend(0, Xui_csr[0])
related = model.similar_items(0)


# In[234]:


related


# In[235]:


itemFactors = model.item_factors
userFactors = model.user_factors


# In[236]:


score = numpy.dot(userFactors[userIDs[allHours[0][0]]], itemFactors[gameIDs[allHours[0][1]]])
score


# In[216]:


m = 0
def BPRPredict(u,g,th=0.9):
    global m
    if u not in userIDs or g not in gameIDs :
        return 0
    score = numpy.dot(userFactors[userIDs[u]], itemFactors[gameIDs[g]])
    if score>m:
        m = score
    return score > th


# In[189]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]


# In[225]:


gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in hoursTrain:
    gameCount[game] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]

mostPopular.sort()
mostPopular.reverse()


# In[226]:


threshold =  2 * totalPlayed / 3

return12 = set()
count2 = 0
for ic, i in mostPopular:
  count2 += ic
  return12.add(i)
  if count2 > threshold: break

def popPredict(u,g):
    if g in return12:
        return 1
    else:
        return 0


# In[227]:


usergames_train,gameusers_train = UsersGames(hoursTrain)


# In[87]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[88]:


def JaccardPredict(u,g):
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


# In[228]:


predictions = [1 if popPredict(u, g) or BPRPredict(u, g, hi) else 0 for u,g,p in augmentedValidationSet]
y = [p for u,d,p in augmentedValidationSet]
accuracy = sum([p == l for p, l in zip(y, predictions)]) / len(augmentedValidationSet)
accuracy


# In[218]:


m


# In[237]:


hacc = 0
hi = 0
for i in range(80,120):
    predictions = [1 if popPredict(u, g) or BPRPredict(u, g, i/100)else 0 for u,g,p in augmentedValidationSet]
    y = [p for u,d,p in augmentedValidationSet]
    accuracy = sum([p == l for p, l in zip(y, predictions)]) / len(augmentedValidationSet)
    print(accuracy)
    if accuracy > hacc:
        hacc = accuracy
        hi = i
hacc, hi


# In[238]:


hacc, hi


# In[ ]:





# In[278]:


def read_csv_file(file_path):
    gamesPerUser = {}
    for l in open(file_path):
        if l.startswith("userID"):
            continue
        u,g = l.strip().split(',')
        if u not in gamesPerUser:
            gamesPerUser[u] = set()
        gamesPerUser[u].add(g) 

    return gamesPerUser


# In[282]:


# predictions = open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/assignment1/halftest/predictions_Played.csv", 'w')
# gamesPerUserTest = read_csv_file("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/pairs_Played.csv")
# for l in open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/pairs_Played.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue
#     u,g = l.strip().split(',')
#     baselinepre = popPredict(u, g)
#     pred0 = 0

#     gamesauser = gamesPerUserTest[u]
#     if len(gamesauser) == 1:
#         baselinepre = popPredict(u, g)
#         bprpre = BPRPredict(u, g, hi/100)
#         if baselinepre or bprpre:
#             pred0 = 1
#         else:
#             pred0 = 0
#     else:
#         gamespopauser = []
#         gamesnotpopauser = []
#         for gu in gamesauser:
#             if not popPredict(u, gu):
#                 gamesnotpopauser.append(gu)
#             else:
#                 gamespopauser.append(gu)
#         if (len(gamespopauser)<= round(len(gamesnotpopauser)/2) and g in gamespopauser):
#             pred0 = 1
#         elif (len(gamespopauser)> round(len(gamesnotpopauser)/2) and g in gamespopauser):
#             gamespopauser_score = {}
#             for gu in gamespopauser:
#                 gamespopauser_score[gu]=BPRPredict(u, gu, hi/100)
#             a = sorted(gamespopauser_score.items(), key=lambda x: x[1], reverse=True)
#             l = len(gamespopauser) - round(len(gamesnotpopauser)/2)
#             if l == 1:
#                 if g == a[0]:
#                     pred0 = 1
#             elif g in a[:l]:
#                     pred0 = 1
#             else:
#                 pred0 = 0
#         elif len(gamesnotpopauser)<= round(len(gamesnotpopauser)/2):
#             pred0 = 0
#         else:
#             gamesnotpopauser_score = {}
#             for gu in gamesnotpopauser:
#                 gamesnotpopauser_score[gu]=BPRPredict(u, gu, hi/100)
#             a = sorted(gamesnotpopauser_score.items(), key=lambda x: x[1], reverse=True)
#             l = len(gamesnotpopauser) - round(len(gamesnotpopauser)/2)
#             if l == 1:
#                 if g == l:
#                     pred0 = 1
#             elif g in a[:l]:
#                 pred0 = 1
#             else:
#                 pred0 = 0
#         pred0 = 0

#     _ = predictions.write(u + ',' + g + ',' + str(pred0) + '\n')
#     i+=1
# predictions.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[283]:


predictions = open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/assignment1/test/predictions_Played.csv", 'w')
for l in open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    baselinepre = popPredict(u, g)
    bprpre = BPRPredict(u, g, hi/100)
    if baselinepre or bprpre:
        pred = 1
    else:
        pred = 0

    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[ ]:





# In[ ]:





# In[240]:


##################################################
# Hours played prediction                        #
##################################################


# In[241]:


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[242]:


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


# In[243]:


betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0


# In[244]:


alpha = globalAverage # Could initialize anywhere, this is a guess


# In[245]:


lambda_ = 1
iter = 100
tolerance = 1e-5 
pmse = 0
converged = False


# In[246]:


usergameHours = {}
for u, g, d in hoursTrain:
    usergameHours[(u,g)] = d['hours_transformed']


# In[247]:


usergames_train,gameusers_train = UsersGames(hoursTrain)


# In[248]:


def iterate(lambu,lambi):
    alpha = globalAverage
    betau_iterate = {}
    betai_iterate = {}
    for u in hoursPerUser:
        betau_iterate[u] = 0
    
    for g in hoursPerItem:
        betai_iterate[g] = 0
    iter = 100
    tolerance = 1e-8
    pmse = float("inf")
    converged = False
    tolerance_count=3
    
    for iteration in range(iter):
        alpha = sum([ usergameHours[d] - betau_iterate[d[0]] - betai_iterate[d[1]] for d in usergameHours])/len(usergameHours)
        
        for u in usergames_train:
            betau_iterate[u] = sum([ usergameHours[(u,g)] - alpha - betai_iterate[g] for g in usergames_train[u]])/(lambu + len(usergames_train[u]))
    
        for g in gameusers_train:
            betai_iterate[g] = sum([ usergameHours[(u,g)] - alpha - betau_iterate[u] for u in gameusers_train[g]])/(lambi + len(gameusers_train[g]))
    
        predictions = [alpha + betau_iterate[u] + betai_iterate[g] for u,g,d in hoursValid]
        y = [d['hours_transformed'] for u,g,d in hoursValid]
    
        mse = mean_squared_error(predictions, y)
        
        if mse > pmse:
            break
        else:
            pmse = mse

    return alpha, betau_iterate, betai_iterate, mse


# In[251]:


# Better lambda...
bestmse=float("inf")
bestlambu = 7.48
bestlambi = 0
for li in range(210,230,1):
    _,_,_,t = iterate(bestlambu,li/100)
    print(t,li)
    if t < bestmse:
        bestmse = t
        bestlambi = li/100
    
bestmse,bestlambi


# In[252]:


# Better lambda...
bestmse=float("inf")
bestlambu = 0
bestlambi = 2.2
for lu in range(740,750,1):
    _,_,_,t = iterate(lu/100,bestlambi)
    print(t,lu)
    if t < bestmse:
        bestmse = t
        bestlambu = lu/100
    
bestmse,bestlambu


# In[ ]:


bestmse,bestlambu


# In[ ]:


bestmse,bestlambi


# In[249]:


alpha, betaU, betaI, mse5 = iterate(7.48,2.2)
mse5


# In[250]:


predictions = open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/assignment1/test/predictions_Hours.csv", 'w')
for l in open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    bu = betaU[u]
    bi = betaI[g]
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()


# In[ ]:




