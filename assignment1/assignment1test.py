#!/usr/bin/env python
# coding: utf-8

# In[390]:


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


# In[367]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[368]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[369]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[370]:


answers = {}


# In[371]:


# Some data structures that will be useful


# In[372]:


allHours = []
for l in readJSON("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/train.json.gz"):
    allHours.append(l)


# In[373]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g,r))
    hoursPerItem[g].append((u,r))


# In[374]:


##################################################
# Play prediction                                #
##################################################


# In[375]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0


# In[376]:


userSet = set()
gameSet = set()
playedSet = set()

for u,g,d in allHours:
    userSet.add(u)
    gameSet.add(g)
    playedSet.add((u,g))

lUserSet = list(userSet)
lGameSet = list(gameSet)

notPlayedValid = set()
for u,g,d in hoursValid:
    g = random.choice(lGameSet)
    while (u,g) in playedSet or (u,g) in notPlayedValid:
        g = random.choice(lGameSet)
    notPlayedValid.add((u,g))

playedValid = set()
for u,g,r in hoursValid:
    playedValid.add((u,g))


# In[377]:


popThreshold = 2/3
simPlayedThreshold = 0.0235


# In[378]:


thpop = [i/30 for i in range(15,30)]
mses = []
for t in thpop:
    mses.append(popTh(t))


# In[ ]:


thjaccards = [1*i/10000 for i in range(230,350)]
mses = []
for t in thjaccards:
    mses.append(popJaccardPredict(2/3,t))


# In[379]:


def popJaccardPredict(popth,jth):
    gameCount = defaultdict(int)
    totalPlayed = 0
    
    for u,g,_ in hoursTrain:
        gameCount[g] += 1
        totalPlayed += 1
    
    mostPopular = [(gameCount[x], x) for x in gameCount]
    mostPopular.sort()
    mostPopular.reverse()
    
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > popth * totalPlayed: break
    
    predictions = 0
    for (label,sample) in [(1, playedValid), (0, notPlayedValid)]:
        for (u,g) in sample:
            maxJaccard = 0
            users = set(hoursPerItem[g])
            for g2,_ in hoursPerUser[u]:
                sim = Jaccard(users,set(hoursPerItem[g2]))
                if sim > maxJaccard:
                    maxJaccard = sim
            if maxJaccard > jth or g in return1:
                pred = 1
            else:
                pred = 0
            if pred == label:
                predictions += 1
    print(str(predictions / (len(playedValid) + len(notPlayedValid)))+" "+str(jth))
    return predictions / (len(playedValid) + len(notPlayedValid))


# In[380]:


popJaccardPredict(popThreshold, simPlayedThreshold)


# In[381]:


predictions = open("predictions_Played.csv", 'w')
for l in open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > popThreshold * totalPlayed: break
    
    maxJaccard = 0
    users = set(hoursPerItem[g])
    for g2,_ in hoursPerUser[u]:
        sim = Jaccard(users,set(hoursPerItem[g2]))
        if sim > maxJaccard:
            maxJaccard = sim
    if maxJaccard > simPlayedThreshold or g in return1:
        pred = 1
    else:
        pred = 0
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[31]:


##################################################
# Hours played prediction                        #
##################################################


# In[382]:


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[383]:


def UsersGames(data):
    usergames = defaultdict(set)
    gameusers = defaultdict(set)
    for u, g, d in data:
        usergames[u].add(g)
        gameusers[g].add(u)
    return usergames, gameusers


# In[384]:


usergames_train,gameusers_train = UsersGames(hoursTrain)


# In[234]:


from random import random
from math import sqrt

# Initialize parameters
num_features = 10  # This is an example, can be tuned

# Latent feature matrices
gamma_u = {u: [random() / sqrt(num_features) for _ in range(num_features)] for u in hoursPerUser}
gamma_i = {g: [random() / sqrt(num_features) for _ in range(num_features)] for g in hoursPerItem}

def predict(u, g):
    dot_product = sum(gamma_u[u][k] * gamma_i[g][k] for k in range(num_features))
    return alpha + betaU.get(u, 0) + betaI.get(g, 0) + dot_product



# In[385]:


betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0


# In[386]:


alpha = globalAverage # Could initialize anywhere, this is a guess


# In[387]:


lambda_ = 1
iter = 100
tolerance = 1e-5 
pmse = 0
converged = False


# In[388]:


usergameHours = {}
for u, g, d in hoursTrain:
    usergameHours[(u,g)] = d['hours_transformed']


# In[249]:


from random import random
from math import sqrt

# Initialize parameters
num_features = 10  # This is an example, can be tuned

# Latent feature matrices
gamma_u = {u: [random() / sqrt(num_features) for _ in range(num_features)] for u in hoursPerUser}
gamma_i = {g: [random() / sqrt(num_features) for _ in range(num_features)] for g in hoursPerItem}

def predict(u, g):
    dot_product = sum(gamma_u[u][k] * gamma_i[g][k] for k in range(num_features))
    return alpha + betaU.get(u, 0) + betaI.get(g, 0) + dot_product

def train(lamb, iterations, learning_rate, tolerance, early_stopping_rounds):
    global alpha, betaU, betaI, gamma_u, gamma_i
    
    last_loss = float('inf')
    best_loss = float('inf')
    loss_increasing_rounds = 0
    learning_rate_schedule = learning_rate
    
    for iteration in range(iterations):
        total_loss = 0
        learning_rate_schedule *= (0.99 ** iteration)  # Gradually decrease learning rate
        
        for (u, g), r in usergameHours.items():
            prediction = predict(u, g)
            error = r - prediction

            # Update biases
            betaU[u] = betaU.get(u, 0) + learning_rate_schedule * (error - lamb * betaU.get(u, 0))
            betaI[g] = betaI.get(g, 0) + learning_rate_schedule * (error - lamb * betaI.get(g, 0))

            # Update latent factors
            for k in range(num_features):
                pu = gamma_u[u][k]
                qi = gamma_i[g][k]

                gamma_u[u][k] = pu + learning_rate_schedule * (error * qi - lamb * pu)
                gamma_i[g][k] = qi + learning_rate_schedule * (error * pu - lamb * qi)
            
            total_loss += error ** 2

        # Regularization term for biases and latent factors
        for u in betaU:
            total_loss += lamb * betaU[u] ** 2
        for g in betaI:
            total_loss += lamb * betaI[g] ** 2
        for u in gamma_u:
            for k in range(num_features):
                total_loss += lamb * gamma_u[u][k] ** 2
        for g in gamma_i:
            for k in range(num_features):
                total_loss += lamb * gamma_i[g][k] ** 2

        print(total_loss
        # Early stopping condition
        if total_loss > last_loss:
            loss_increasing_rounds += 1
            if loss_increasing_rounds >= early_stopping_rounds:
                print(f'Stopping early at iteration {iteration}')
                break
        else:
            loss_increasing_rounds = 0
            if total_loss < best_loss:
                best_loss = total_loss
                # Save the best parameters if necessary
        
        if abs(last_loss - total_loss) < tolerance:
            print(f'Converged at iteration {iteration}')
            break
        
        last_loss = total_loss
    
    return betaU, betaI, gamma_u, gamma_i, best_loss



# Initialize the biases
betaU = {u: 0 for u in hoursPerUser}
betaI = {g: 0 for g in hoursPerItem}
alpha = globalAverage  # This is the global average we calculated earlier

# Run training
train(lamb=0.1, iterations=1000, learning_rate=0.005, tolerance=1e-4, early_stopping_rounds=5)

predictions = [predict(u,g) for u,g,d in hoursValid]
y = [d['hours_transformed'] for u,g,d in hoursValid]

mse = mean_squared_error(predictions, y)
mse


# In[393]:


from random import random
from math import sqrt

# Initialize parameters
num_features = 25  # This is an example, can be tuned

# Latent feature matrices
gamma_u = {u: [random() / sqrt(num_features) for _ in range(num_features)] for u in hoursPerUser}
gamma_i = {g: [random() / sqrt(num_features) for _ in range(num_features)] for g in hoursPerItem}

# Predict function
def predict(u, g):
    dot_product = sum(gamma_u[u][k] * gamma_i[g][k] for k in range(num_features))
    return alpha + betaU.get(u, 0) + betaI.get(g, 0) + dot_product

# Function to calculate the loss on validation set
def calculate_validation_loss(validation_data, lamb):
    validation_loss = 0
    for u, g, r in validation_data:
        prediction = predict(u, g)
        error = r - prediction
        validation_loss += error ** 2
    
    # Regularization term
    for u in betaU:
        validation_loss += lamb * betaU[u] ** 2
    for g in betaI:
        validation_loss += lamb * betaI[g] ** 2
    for u in gamma_u:
        for k in range(num_features):
            validation_loss += lamb * gamma_u[u][k] ** 2
    for g in gamma_i:
        for k in range(num_features):
            validation_loss += lamb * gamma_i[g][k] ** 2
    
    return validation_loss / len(validation_data)

# Training function
def train(lamb, iterations, initial_learning_rate, tolerance, early_stopping_rounds, validation_data):
    global alpha, betaU, betaI, gamma_u, gamma_i

    learning_rate = initial_learning_rate
    best_loss = float('inf')
    no_improvement_count = 0
    best_betaU, best_betaI, best_gamma_u, best_gamma_i = {}, {}, {}, {}
    
    for iteration in range(iterations):
        total_loss = 0

        learning_rate = learning_rate / (1 + decay_rate * iteration)

        for (u, g), r in usergameHours.items():
            prediction = predict(u, g)
            error = r - prediction

            


            # Update biases
            betaU[u] = betaU.get(u, 0) + learning_rate * (error - lamb * betaU.get(u, 0))
            betaI[g] = betaI.get(g, 0) + learning_rate * (error - lamb * betaI.get(g, 0))

            # Update latent factors
            for k in range(num_features):
                pu = gamma_u[u][k]
                qi = gamma_i[g][k]

                gamma_u[u][k] = pu + learning_rate * (error * qi - lamb * pu)
                gamma_i[g][k] = qi + learning_rate * (error * pu - lamb * qi)
            
            total_loss += error ** 2

        # Regularization term for biases and latent factors
        for u in betaU:
            total_loss += lamb * betaU[u] ** 2
        for g in betaI:
            total_loss += lamb * betaI[g] ** 2
        for u in gamma_u:
            for k in range(num_features):
                total_loss += lamb * gamma_u[u][k] ** 2
        for g in gamma_i:
            for k in range(num_features):
                total_loss += lamb * gamma_i[g][k] ** 2
        
        validation_loss = calculate_validation_loss(validation_data, lamb)

        print(best_loss)
        # Check if validation loss improved
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_betaU, best_betaI, best_gamma_u, best_gamma_i = betaU.copy(), betaI.copy(), gamma_u.copy(), gamma_i.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print(f'Stopping early at iteration {iteration}')
            break

        if abs(total_loss - validation_loss) < tolerance:
            print(f'Converged at iteration {iteration}')
            break
    
    return best_betaU, best_betaI, best_gamma_u, best_gamma_i, best_loss

# Initialize the biases
alpha = globalAverage
betaU = {u: random() * 0.1 for u in hoursPerUser}
betaI = {g: random() * 0.1 for g in hoursPerItem}

# Validation data
validation_data = [(u, g, d['hours_transformed']) for u, g, d in hoursValid]

decay_rate = 0.001
# Run training with early stopping
best_betaU, best_betaI, best_gamma_u, best_gamma_i, best_loss = train(
    lamb=0.01, 
    iterations=1000, 
    initial_learning_rate=0.003, 
    tolerance=1e-5, 
    early_stopping_rounds=5, 
    validation_data=validation_data
)

print(f'Best validation loss: {best_loss}')
#0.001, 0.003 3.0531


# In[356]:


predictions = [predict(u,g) for u,g,d in hoursValid]
y = [d['hours_transformed'] for u,g,d in hoursValid]

mse = mean_squared_error(predictions, y)
mse


# In[353]:


def train_and_evaluate(num_features, lamb, initial_learning_rate, iterations, tolerance, early_stopping_rounds, train_data, validation_data):
    global alpha, betaU, betaI, gamma_u, gamma_i
    
    # Initialize the latent factors with the new number of features
    alpha = sum([r for _, _, r in train_data]) / len(train_data)
    betaU = {u: random() * 0.1 for u, _, _ in train_data}
    betaI = {g: random() * 0.1 for _, g, _ in train_data}
    gamma_u = {u: [random() / sqrt(num_features) for _ in range(num_features)] for u, _, _ in train_data}
    gamma_i = {g: [random() / sqrt(num_features) for _ in range(num_features)] for _, g, _ in train_data}
    
    # Train the model
    best_betaU, best_betaI, best_gamma_u, best_gamma_i, best_loss = train(
        lamb=lamb, 
        iterations=iterations, 
        initial_learning_rate=initial_learning_rate, 
        tolerance=tolerance, 
        early_stopping_rounds=early_stopping_rounds, 
        validation_data=validation_data
    )
    
    # Calculate the loss on the validation set
    validation_loss = calculate_validation_loss(validation_data, lamb)
    
    return validation_loss

# Define a range of feature counts to try
feature_counts = [5, 10, 15, 20, 25, 30]
validation_losses = []

# Iterate over the range of feature counts
for num_features in feature_counts:
    print(f"Evaluating model with {num_features} features...")
    loss = train_and_evaluate(
        num_features=num_features,
        lamb=0.01,
        initial_learning_rate=0.003,
        iterations=1000,
        tolerance=1e-5,
        early_stopping_rounds=20,
        train_data=[(u, g, d['hours_transformed']) for u, g, d in hoursTrain],  # Replace with your actual training data
        validation_data=[(u, g, d['hours_transformed']) for u, g, d in hoursValid]  # Replace with your actual validation data
    )
    validation_losses.append((num_features, loss))

# Select the number of features with the lowest validation loss
best_num_features, best_loss = min(validation_losses, key=lambda x: x[1])
print(f'Best number of features: {best_num_features} with validation loss: {best_loss}')


# In[395]:


#cross valid
from sklearn.model_selection import KFold
import numpy as np
import random

def cross_validate(num_features, lamb, learning_rate, iterations, tolerance, early_stopping_rounds, train_data):
    kf = KFold(n_splits=5)  # 5-fold cross-validation
    fold_validation_losses = []

    for train_index, test_index in kf.split(train_data):
        train_subset = [train_data[i] for i in train_index]
        validation_subset = [train_data[i] for i in test_index]

        # Initialize the latent factors with the new number of features
        alpha = sum([r for _, _, r in train_subset]) / len(train_subset)
        betaU = {u: random.random() * 0.1 for u, _, _ in train_subset}
        betaI = {g: random.random() * 0.1 for _, g, _ in train_subset}
        gamma_u = {u: [random.random() / sqrt(num_features) for _ in range(num_features)] for u, _, _ in train_subset}
        gamma_i = {g: [random.random() / sqrt(num_features) for _ in range(num_features)] for _, g, _ in train_subset}
    
        # Train the model
        best_betaU, best_betaI, best_gamma_u, best_gamma_i, best_loss = train(
            lamb=lamb, 
            iterations=iterations, 
            initial_learning_rate=learning_rate, 
            tolerance=tolerance, 
            early_stopping_rounds=early_stopping_rounds, 
            validation_data=validation_subset
        )
    
        # Calculate the loss on the validation subset
        validation_loss = calculate_validation_loss(validation_subset, lamb)
        fold_validation_losses.append(validation_loss)

    average_validation_loss = np.mean(fold_validation_losses)
    return average_validation_loss

# Define a range of feature counts to try
feature_counts = [25]
average_validation_losses = []

# Perform cross-validation on the training set
for num_features in feature_counts:
    print(f"Evaluating model with {num_features} features...")
    avg_loss = cross_validate(
        num_features=num_features,
        lamb=0.01,
        learning_rate=0.003,
        iterations=1000,
        tolerance=1e-5,
        early_stopping_rounds=5,
        train_data=[(u, g, d['hours_transformed']) for u, g, d in hoursTrain]  # Replace with your actual training data
    )
    average_validation_losses.append((num_features, avg_loss))

# Select the number of features with the lowest average validation loss
best_num_features, best_avg_loss = min(average_validation_losses, key=lambda x: x[1])
print(f'Best number of features: {best_num_features} with average validation loss: {best_avg_loss}')

# Retrain the model with the best number of features on the entire training set
# and evaluate on the separate validation set
final_validation_loss = train_and_evaluate(
    num_features=best_num_features,
    lamb=0.01,
    learning_rate=0.003,
    iterations=1000,
    tolerance=1e-5,
    early_stopping_rounds=5,
    train_data=[(u, g, d['hours_transformed']) for u, g, d in hoursTrain],  # Replace with your entire training data
    validation_data=[(u, g, d['hours_transformed']) for u, g, d in hoursValid]  # Replace with your actual validation data
)

print(f'Final validation loss with {best_num_features} features: {final_validation_loss}')


# In[396]:


predictions = [predict(u,g) for u,g,d in hoursValid]
y = [d['hours_transformed'] for u,g,d in hoursValid]

mse = mean_squared_error(predictions, y)
mse


# In[398]:


def predictcross(u, g, alpha, betaU, betaI, gamma_u, gamma_i):
    """Predicts the hours played using the model parameters."""
    user_factor = gamma_u.get(u, [0]*len(gamma_u[list(gamma_u.keys())[0]]))
    item_factor = gamma_i.get(g, [0]*len(gamma_i[list(gamma_i.keys())[0]]))
    dot_product = sum(user_factor[k] * item_factor[k] for k in range(len(user_factor)))
    return alpha + betaU.get(u, 0) + betaI.get(g, 0) + dot_product

def calculate_mse(validation_data, alpha, betaU, betaI, gamma_u, gamma_i):
    """Calculates the mean squared error on the validation set."""
    errors = []
    for u, g, actual_hours in validation_data:
        predicted_hours = predictcross(u, g, alpha, betaU, betaI, gamma_u, gamma_i)
        errors.append((predicted_hours - actual_hours) ** 2)
    mse = sum(errors) / len(errors)
    return mse


mse = calculate_mse([(u, g, d['hours_transformed']) for u, g, d in hoursValid], alpha, betaU, betaI, gamma_u, gamma_i)
print("MSE on validation set:", mse)


# In[311]:


hoursTrain[0]


# In[229]:


betaU, betaI, mse2 = iterate(2)
mse2


# In[248]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    re = predict(u,g)
    
    _ = predictions.write(u + ',' + g + ',' + str(re) + '\n')

predictions.close()


# In[36]:


alpha = globalAverage # Could initialize anywhere, this is a guess


# In[37]:


def iterate(lamb):
    newAlpha = 0
    for u,g,d in hoursTrain:
        r = d['hours_transformed']
        newAlpha += r - (betaU[u] + betaI[g])
    alpha = newAlpha / len(hoursTrain)
    for u in hoursPerUser:
        newBetaU = 0
        for g,r in hoursPerUser[u]:
            newBetaU += r - (alpha + betaI[g])
        betaU[u] = newBetaU / (lamb + len(hoursPerUser[u]))
    for g in hoursPerItem:
        newBetaI = 0
        for u,r in hoursPerItem[g]:
            newBetaI += r - (alpha + betaU[u])
        betaI[g] = newBetaI / (lamb + len(hoursPerItem[g]))
    mse = 0
    for u,g,d in hoursTrain:
        r = d['hours_transformed']
        prediction = alpha + betaU[u] + betaI[g]
        mse += (r - prediction)**2
    regularizer = 0
    for u in betaU:
        regularizer += betaU[u]**2
    for g in betaI:
        regularizer += betaI[g]**2
    mse /= len(hoursTrain)
    return mse, mse + lamb*regularizer


# In[38]:


mse,objective = iterate(1)
newMSE,newObjective = iterate(1)
iterations = 2


# In[39]:


while iterations < 10 or objective - newObjective > 0.01:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(1)
    iterations += 1
    print("Objective after "
        + str(iterations) + " iterations = " + str(newObjective))
    print("MSE after "
        + str(iterations) + " iterations = " + str(newMSE))


# In[40]:


validMSE = 0
for u,g,d in hoursValid:
    r = d['hours_transformed']
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if g in betaI:
        bi = betaI[g]
    prediction = alpha + bu + bi
    validMSE += (r - prediction)**2

validMSE /= len(hoursValid)
print("Validation MSE = " + str(validMSE))


# In[41]:


answers['Q6'] = validMSE


# In[42]:


assertFloat(answers['Q6'])


# In[43]:


### Question 7


# In[44]:


betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')


# In[45]:


answers['Q7'] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]


# In[46]:


answers['Q7']


# In[47]:


assertFloatList(answers['Q7'], 4)


# In[48]:


### Question 8


# In[49]:


# Better lambda...

iterations = 1
while iterations < 10 or objective - newObjective > 0.01:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(5)
    iterations += 1
    print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
    print("MSE after " + str(iterations) + " iterations = " + str(newMSE))


# In[50]:


alpha_ = alpha


# In[51]:


validMSE = 0
for u,g,d in hoursValid:
    r = d['hours_transformed']
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if g in betaI:
        bi = betaI[g]
    prediction = alpha + bu + bi
    validMSE += (r - prediction)**2

validMSE /= len(hoursValid)
print("Validation MSE = " + str(validMSE))


# In[52]:


answers['Q8'] = (5.0, validMSE)


# In[53]:


assertFloatList(answers['Q8'], 2)


# In[54]:


predictions = open("HWpredictions_Hours.csv", 'w')
for l in open("/home/julian/Downloads/assignment1/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if g in betaI:
        bi = betaI[g]
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()


# In[55]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[1]:


import gzip
import random
import scipy
import tensorflow as tf
from collections import defaultdict
from implicit import bpr
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split


# In[2]:


pip install --upgrade tensorflow


# In[2]:


pip install implicit


# In[13]:


pip install --use-pep517 surprise


# In[6]:


pip install wheel setuptools pip --upgrade


# In[ ]:




