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
