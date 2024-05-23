#!/usr/bin/env python
# coding: utf-8

# In[27]:


pip install scikit-learn


# In[17]:


pip install sklearn


# In[385]:


import random
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip
import math


# In[386]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[387]:


answers = {}


# In[388]:


def parseData(fname):
    for l in open(fname):
        yield eval(l)


# In[389]:


data = list(parseData("beer_50000.json"))


# In[390]:


random.seed(0)
random.shuffle(data)


# In[391]:


dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]


# In[392]:


maxLen  = max([len(d['review/text']) for d in dataTrain])


# In[393]:


yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]


# In[394]:


dataTrain[0]


# In[395]:


categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1


# In[396]:


categories = [c for c in categoryCounts if categoryCounts[c] > 1000]


# In[397]:


catID = dict(zip(list(categories),range(len(categories))))


# In[398]:


def myonehotcat(i):
    t = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    if i:
        t[i] = 1
    return t 


# In[399]:


def myFeat(d, incat, inrev, inlen):
    if incat and not inrev and not inlen:
        t = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        i = catID.get(d['beer/style'])
        if i:
            t[i] = 1

    elif not incat and inrev and inlen:
        t = [0,0,0,0,0,  0]
        t[0] = d['review/appearance']
        t[1] = d['review/palate']
        t[2] = d['review/taste']
        t[3] = d['review/overall']
        t[4] = d['review/aroma']
        t[5] = len(d['review/text'])/maxLen

    elif incat and not inrev and inlen:
        t = [0,0,0,0,0,0,0,0,0,0,0,0,0,  0]

        i = catID.get(d['beer/style'])
        if i:
            t[i] = 1

        t[13] = len(d['review/text'])/maxLen

    elif incat and inrev and not inlen:
        t = [0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0]

        i = catID.get(d['beer/style'])
        if i:
            t[i] = 1

        t[13] = d['review/appearance']
        t[14] = d['review/palate']
        t[15] = d['review/taste']
        t[16] = d['review/overall']
        t[17] = d['review/aroma']

    elif incat and inrev and inlen:
        t = [0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,  0]

        i = catID.get(d['beer/style'])
        if i:
            t[i] = 1

        t[13] = d['review/appearance']
        t[14] = d['review/palate']
        t[15] = d['review/taste']
        t[16] = d['review/overall']
        t[17] = d['review/aroma']

        t[18] = len(d['review/text'])/maxLen
        
    return t 


# In[400]:


def myber(pred, y):
    TP = sum([(p and l) for (p,l) in zip(pred, y)])
    TN = sum([(not p and not l) for (p,l) in zip(pred, y)])
    FP = sum([(p and not l) for (p,l) in zip(pred, y)])
    FN = sum([(not p and l) for (p,l) in zip(pred, y)])

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    BER = 1 - 1/2 * (TPR + TNR)

    return BER


# In[401]:


def feat(includeCat = True, includeReview = True, includeLength = True):

    featX =  [myFeat(d, includeCat, includeReview, includeLength) for d in dataTrain]
    featXV = [myFeat(d, includeCat, includeReview, includeLength) for d in dataValid]
    featXT = [myFeat(d, includeCat, includeReview, includeLength) for d in dataTest]

    
    return featX, featXV, featXT


# In[402]:


def pipeline(reg, includeCat = True, includeReview = True, includeLength = True):
    featX, featXV, featXT = feat(includeCat, includeReview, includeLength)

    mod = sklearn.linear_model.LogisticRegression(C = reg, class_weight='balanced')
    mod.fit(featX,yTrain)
    predictionsV = mod.predict(featXV)
    predictionsT = mod.predict(featXT)

    return mod, myber(predictionsV, yValid), myber(predictionsT, yTest)


# In[403]:


### Question 1


# In[404]:


mod, validBER, testBER = pipeline(10, True, False, False)
validBER, testBER


# In[405]:


answers['Q1'] = [validBER, testBER]


# In[406]:


assertFloatList(answers['Q1'], 2)


# In[407]:


### Question 2


# In[408]:


mod, validBER, testBER = pipeline(10, True, True, True)
validBER, testBER


# In[409]:


answers['Q2'] = [validBER, testBER]


# In[410]:


assertFloatList(answers['Q2'], 2)


# In[411]:


### Question 3


# In[412]:


dataTrain


# In[413]:


minvber = 1
bestc = 0
for c in [0.001, 0.01, 0.1, 1, 10]:
    mod, validBER, testBER = pipeline(c, True, True, True)
    if validBER < minvber:
        minvber = validBER
        bestc = c
    print(str(c) +":"+  str(validBER) +" "+ str(testBER))
bestc


# In[414]:


bestC = bestc


# In[415]:


mod, validBER, testBER = pipeline(bestC, True, True, True)


# In[416]:


answers['Q3'] = [bestC, validBER, testBER]


# In[417]:


assertFloatList(answers['Q3'], 3)


# In[418]:


### Question 4


# In[419]:


mod, validBER, testBER_noCat = pipeline(1, False, True, True)


# In[420]:


mod, validBER, testBER_noReview = pipeline(1, True, False, True)


# In[421]:


mod, validBER, testBER_noLength = pipeline(1, True, True, False)


# In[425]:


answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]
testBER_noCat, testBER_noReview, testBER_noLength


# In[426]:


assertFloatList(answers['Q4'], 3)


# In[424]:


### Question 5


# In[428]:


path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')


# In[429]:


header


# In[430]:


dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)


# In[431]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]
dataTrain


# In[432]:


# Feel free to keep or discard

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair
reviewsPerUser = defaultdict(list)

for d in dataTrain:
    user,item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user,item)] = d['star_rating']
    itemNames[item] = d['product_title']


# In[433]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)

ratingMean = sum([d['star_rating'] for d in dataset]) / len(dataset)


# In[434]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[435]:


def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[436]:


query = 'B00KCHRKD6'


# In[437]:


ms = mostSimilar(query, 10)
ms


# In[438]:


answers['Q5'] = ms


# In[439]:


assertFloatList([m[0] for m in ms], 10)


# In[440]:


### Question 6


# In[441]:


def MSE(y, ypred):
    return sum([(p - l)**2 for (p,l) in zip(y, ypred)])/len(y)


# In[442]:


def predictRating(user,item):
    numerator = 0
    denominator = 0
    
    # If the item hasn't been seen before, return the global average rating
    if item not in itemAverages:
        return ratingMean

    # Calculate predicted rating based on the formula
    for j in itemsPerUser[user]:  # Iterating over items that the user has rated
        if j == item:
            continue
        similarity = Jaccard(usersPerItem[item], usersPerItem[j])
        numerator += (ratingDict[(user, j)] - itemAverages[j]) * similarity
        denominator += similarity

    # If no similar items exist (denominator is zero), return the item's average rating
    if denominator == 0:
        return itemAverages[item]

    return itemAverages[item] + numerator / denominator


# In[443]:


simPredictions = [predictRating(d['customer_id'], d['product_id']) for d in dataTest] 


# In[444]:


labels = [d['star_rating'] for d in dataTest]


# In[445]:


answers['Q6'] = MSE(simPredictions, labels)
answers['Q6']


# In[446]:


assertFloat(answers['Q6'])


# In[ ]:


### Question 7


# In[369]:


from datetime import datetime

usersPerItemT = defaultdict(set) # Maps an item to the users who rated it
itemsPerUserT = defaultdict(set) # Maps a user to the items that they rated
date_format = "%Y-%m-%d"
ratingDictTime = {}

for d in dataTrain:
    user,item = d['customer_id'], d['product_id']
    usersPerItemT[item].add(user)
    itemsPerUserT[user].add(item)

    timestamp_dt = datetime.strptime(d['review_date'], date_format)
    
    ratingDictTime[(user, item)] = {
        'rating': d['star_rating'],
        'timestamp': timestamp_dt
    }


# In[370]:


userAveragesT = {}
itemAveragesT = {}

for u in itemsPerUserT:
    rs = [ratingDictTime[(u,i)]['rating'] for i in itemsPerUserT[u]]
    userAveragesT[u] = sum(rs) / len(rs)
    
for i in usersPerItemT:
    rs = [ratingDictTime[(u,i)]['rating'] for u in usersPerItemT[i]]
    itemAveragesT[i] = sum(rs) / len(rs)

ratingMeanT = sum([d['star_rating'] for d in dataset]) / len(dataset)


# In[379]:


def myPridictRating(user, item, mylambda):
    numerator = 0
    denominator = 0

    if item not in itemAveragesT:
        return ratingMeanT

    if user not in usersPerItemT:
        return ratingMeanT
    
    # Calculate predicted rating based on the formula
    for j in itemsPerUser[user]:  # Iterating over items that the user has rated
        if j == item:
            continue
            
        similarity = Jaccard(usersPerItemT[item], usersPerItemT[j])
        
        # Calculate time difference in days
        time_difference = abs((ratingDictTime[(user, item)]['timestamp'] - ratingDictTime[(user, j)]['timestamp']).days)
        decay_factor = math.exp(-mylambda * time_difference)
           
        numerator += (ratingDictTime[(user, j)]['rating'] - userAveragesT[user]) * sim_ij * decay_factor
        denominator += abs(sim_ij * decay_factor)
        
    if denominator == 0:
        if item in itemAveragesT:
            return itemAveragesT[item]
        return ratingMeanT
        
    return userAveragesT[user] + numerator / denominator


# In[383]:


pd = [myPridictRating(d['customer_id'], d['product_id'], 10) for d in dataTest]


# In[384]:


bestmse = MSE(pd, labels)
bestmse


# In[331]:


itsMSE = bestmse


# In[332]:


answers['Q7'] = ["Convert the time string to a datetime object and calculate the difference in days between the two as f(tu,j)", itsMSE]


# In[333]:


assertFloat(answers['Q7'][1])


# In[427]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




