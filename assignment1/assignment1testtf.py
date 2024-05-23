#!/usr/bin/env python
# coding: utf-8

# In[11]:


pip install tensorflow


# In[9]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import tensorflow as tf


# In[2]:


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


# In[3]:


userIDs = {}
itemIDs = {}
interactions = []

for d in parse("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/train.json.gz"):
    u = d['userID']
    i = d['gameID']
    r = d['hours_transformed']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
    interactions.append((u,i,r))


# In[4]:


random.shuffle(interactions)
len(interactions)


# In[5]:


nTrain = int(len(interactions) * 0.9)
nTest = len(interactions) - nTrain
interactionsTrain = interactions[:nTrain]
interactionsTest = interactions[nTrain:]


# In[6]:


itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for u,i,r in interactionsTrain:
    itemsPerUser[u].append(i)
    usersPerItem[i].append(u)


# In[45]:


mu = sum([r for _,_,r in interactionsTrain]) / len(interactionsTrain)


# In[118]:


optimizer = tf.keras.optimizers.Adam(0.1)


# In[119]:


class LatentFactorModel(tf.keras.Model):
    def __init__(self, mu, K, lamb):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i] +\
            tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU**2) +\
                            tf.reduce_sum(self.betaI**2) +\
                            tf.reduce_sum(self.gammaU**2) +\
                            tf.reduce_sum(self.gammaI**2))
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i +\
               tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)


# In[120]:


modelLFM = LatentFactorModel(mu, 35, 0.00001)


# In[121]:


def trainingStep(model, interactions):
    Nsamples = 100000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR = [], [], []
        for _ in range(Nsamples):
            u,i,r = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleR.append(r)

        loss = model(sampleU,sampleI,sampleR)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


# In[ ]:


for i in range(500):
    obj = trainingStep(modelLFM, interactionsTrain)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))


# In[115]:


u,i,r = interactionsTest[0]
u,i,r


# In[116]:


modelLFM.predict(userIDs[u], itemIDs[i]).numpy()


# In[117]:


def calculate_mse(model, interactions):
    """
    Calculate the mean squared error for the given interactions.

    Args:
    model: The LatentFactorModel instance.
    interactions: A list of tuples (user, item, rating).

    Returns:
    float: The mean squared error.
    """
    total_squared_error = 0
    count = 0

    for u, i, r in interactions:
        # Convert user and item to the indices used in the model
        u_index = userIDs[u]
        i_index = itemIDs[i]

        # Predict the rating
        predicted_rating = model.predict(u_index, i_index)
        
        # Compute squared error
        squared_error = tf.square(predicted_rating - r)
        total_squared_error += squared_error
        count += 1

    # Compute mean squared error
    mse = total_squared_error / count
    return mse.numpy()  # Convert to a regular Python number

# Example usage
mse_test = calculate_mse(modelLFM, interactionsTest)
print(f"Test MSE: {mse_test}")


# In[ ]:





# In[ ]:





# In[15]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[16]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[17]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[18]:


answers = {}


# In[19]:


# Some data structures that will be useful


# In[20]:


allHours = []
for l in readJSON("/Users/zhiqiaogong/Projects/JupyterNotebook/cse258/hw3/train.json.gz"):
    allHours.append(l)


# In[21]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g,r))
    hoursPerItem[g].append((u,r))


# In[22]:


##################################################
# Play prediction                                #
##################################################


# In[23]:


#bpr


# In[24]:


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


# In[42]:


user_ids = {u: i for i, u in enumerate(userSet)}
game_ids = {g: i for i, g in enumerate(gameSet)}

num_users = len(user_ids)
num_games = len(game_ids)

# Initialize the user-item matrix with zeros
user_item_matrix = np.zeros((num_users, num_games))

for u, g, _ in hoursTrain:
    user_item_matrix[user_ids[u], game_ids[g]] = 1

# BPR Model
class BPR:
    def __init__(self, num_users, num_items, num_factors=10):
        self.user_factors = np.random.normal(0, 0.1, (num_users, num_factors))
        self.item_factors = np.random.normal(0, 0.1, (num_items, num_factors))
    
    def predict(self, user, item):
        return np.dot(self.user_factors[user], self.item_factors[item])
    
    def train(self, matrix, epochs=100, learning_rate=0.01, lambda_reg=0.01):
         # Initialize Adam optimizer parameters
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m_user, v_user = np.zeros(self.user_factors.shape), np.zeros(self.user_factors.shape)
        m_item, v_item = np.zeros(self.item_factors.shape), np.zeros(self.item_factors.shape)
        t = 0
        for _ in range(epochs):
            t += 1
            user = np.random.randint(num_users)
            pos_items = np.where(matrix[user] == 1)[0]
            neg_items = self.get_neg_item_candidates(user, matrix, pos_items)

            if not neg_items:
                continue

            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(neg_items)

            
            
            # Calculate error and gradients
            x_uij = self.predict(user, pos_item) - self.predict(user, neg_item)
            exp_x = np.exp(-x_uij)
            loss = exp_x / (1 + exp_x)

            user_gradient = (self.item_factors[pos_item] - self.item_factors[neg_item]) * loss - lambda_reg * self.user_factors[user]
            item_pos_gradient = self.user_factors[user] * loss - lambda_reg * self.item_factors[pos_item]
            item_neg_gradient = -self.user_factors[user] * loss - lambda_reg * self.item_factors[neg_item]

            # Update user and item factors using Adam optimizer
            m_user = beta1 * m_user + (1 - beta1) * user_gradient
            v_user = beta2 * v_user + (1 - beta2) * (user_gradient ** 2)
            m_item[pos_item] = beta1 * m_item[pos_item] + (1 - beta1) * item_pos_gradient
            v_item[pos_item] = beta2 * v_item[pos_item] + (1 - beta2) * (item_pos_gradient ** 2)
            m_item[neg_item] = beta1 * m_item[neg_item] + (1 - beta1) * item_neg_gradient
            v_item[neg_item] = beta2 * v_item[neg_item] + (1 - beta2) * (item_neg_gradient ** 2)

            m_user_corr = m_user / (1 - beta1 ** t)
            v_user_corr = v_user / (1 - beta2 ** t)
            m_item_corr = m_item / (1 - beta1 ** t)
            v_item_corr = v_item / (1 - beta2 ** t)

            self.user_factors -= learning_rate * m_user_corr / (np.sqrt(v_user_corr) + epsilon)
            self.item_factors[pos_item] -= learning_rate * m_item_corr[pos_item] / (np.sqrt(v_item_corr[pos_item]) + epsilon)
            self.item_factors[neg_item] -= learning_rate * m_item_corr[neg_item] / (np.sqrt(v_item_corr[neg_item]) + epsilon)

    def get_neg_item_candidates(self, user, matrix, pos_items):
        # Implement a logic to select negative item candidates
        # Example: Choose items not interacted with by the user and are popular among other users
        # Return a list of item indices
        all_items = set(range(matrix.shape[1]))
        neg_candidates = list(all_items - set(pos_items))
        return neg_candidates

# Train the BPR model
bpr_model = BPR(num_users, num_games)
bpr_model.train(user_item_matrix)

# Example prediction (You can replace these with actual user and game IDs from your data)
user_id = 0  # example user index
game_id = 10  # example game index
prediction = bpr_model.predict(user_id, game_id)
print("Prediction score:", prediction)


# In[43]:


# Function to predict for a user-item pair
def predict_bpr(bpr_model, user_id, item_id):
    return bpr_model.predict(user_ids[user_id], game_ids[item_id])

# Calculate predictions for the validation set and compute MSE
def calculate_mse(bpr_model, played_valid, not_played_valid):
    mse = 0
    count = 0

    # For pairs in playedValid, the actual interaction is 1
    for (u, g) in played_valid:
        predicted_score = predict_bpr(bpr_model, u, g)
        mse += (predicted_score - 1) ** 2  # actual interaction is 1
        count += 1

    # For pairs in notPlayedValid, the actual interaction is 0
    for (u, g) in not_played_valid:
        predicted_score = predict_bpr(bpr_model, u, g)
        mse += (predicted_score - 0) ** 2  # actual interaction is 0
        count += 1

    return mse / count if count > 0 else 0

# Calculate MSE on the validation set
mse_validation = calculate_mse(bpr_model, playedValid, notPlayedValid)
print("MSE on Validation Set:", mse_validation)


# In[44]:


def calculate_accuracy(bpr_model, played_valid, not_played_valid, threshold=0.5):
    correct_predictions = 0
    total_predictions = len(played_valid) + len(not_played_valid)

    # Check predictions for playedValid (should be predicted as played, i.e., score > threshold)
    for (u, g) in played_valid:
        predicted_score = predict_bpr(bpr_model, u, g)
        if predicted_score > threshold:
            correct_predictions += 1

    # Check predictions for notPlayedValid (should be predicted as not played, i.e., score <= threshold)
    for (u, g) in not_played_valid:
        predicted_score = predict_bpr(bpr_model, u, g)
        if predicted_score <= threshold:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

# Calculate Accuracy on the validation set
accuracy_validation = calculate_accuracy(bpr_model, playedValid, notPlayedValid)
print("Accuracy on Validation Set:", accuracy_validation)


# In[45]:


# List of hyperparameters to try
factor_options = [10, 20, 30]  # Number of latent factors
learning_rate_options = [0.005, 0.01, 0.05]
regularization_options = [0.001, 0.01, 0.1]

best_accuracy = 0
best_params = {}

for factors in factor_options:
    for lr in learning_rate_options:
        for reg in regularization_options:
            # Initialize and train the BPR model
            bpr_model = BPR(num_users, num_games, num_factors=factors)
            bpr_model.train(user_item_matrix, learning_rate=lr, lambda_reg=reg)

            print(current_accuracy)
            # Evaluate the model
            current_accuracy = calculate_accuracy(bpr_model, playedValid, notPlayedValid)
            
            # Update best parameters if current model is better
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_params = {'factors': factors, 'learning_rate': lr, 'regularization': reg}

print("Best Accuracy:", best_accuracy)
print("Best Hyperparameters:", best_params)


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




