# Map string ID to integer ID
uniq_users = list(set([d['userID'] for d in data_train]))
uniq_items = list(set([d['businessID'] for d in data_train]))
num_users = len(uniq_users)
num_items = len(uniq_items)
userIds = dict(zip(uniq_users, range(num_users)))
itemIds = dict(zip(uniq_items, range(num_items)))
def feat(d):
  uid = d['userID']
  bid = d['businessID']
  return [userIds[uid], itemIds[bid]]
def predict(x,b_u, b_i, alpha):
  pred = np.zeros(len(x))
  for j in range(len(x)):
    u,i = x[j]
    pred[j] = alpha + b_u[u] + b_i[i]
  return pred
# Build rating matrix, businesses visited by user and users that visited a business
R = np.zeros((num_users, num_items))
nums = np.zeros((num_users, num_items))
userToItem = defaultdict(list)
itemToUser = defaultdict(list)
for d in data_train:
  u = userIds[d['userID']]
  i = itemIds[d['businessID']]
  R[u][i] = d['rating']
  nums[u][1] += 1
  userToItem[u].append(i)
  itemToUser[i].append(u)

nums[nums == 0] = 1
# Get the average R for the user if he has visited more than once...
R /= nums 
# Gradient descent based on the lecture notes
def gradient_descent(x, y, b_u, b_i, alpha, max_its, reg):
  preds = predict(x, b_u, b_i, alpha)
  mse = [mean_squared_error(preds, y)]
  for it in range(max_its):
    for j in range(len(x)):
      u = x[j][0]
      i = x[j][1]
      
      # Update rules from lecture
      # Update for user
      userItems = userToItem[u]
      b_u[u] = (- alpha * len(userItems) - b_i[userItems].sum() + R[u,userItems].sum()) / (reg + len(userItems))
      
      # Update for item
      itemUsers = itemToUser[i]
      b_i[i] = (- alpha * len(itemUsers) - b_u[itemUsers].sum() + R[itemUsers, i].sum()) / (reg + len(itemUsers))
      
    # Calculate MSE 
    preds = predict(x, b_u, b_i, alpha)
    mse.append(mean_squared_error(preds, y))
  return alpha, b_i, b_u, mse
x_train = np.array([feat(d) for d in data_train])
y_train = np.array([d['rating'] for d in data_train])
# Random initialize variables
b_u = np.random.random((num_users,))
b_i = np.random.random((num_items))
alpha = y_train.mean()
def predict_test(d, b_u, b_i, alpha):
  try:
    u = userIds[d['userID']]
    i = itemIds[d['businessID']]
    return alpha + b_u[u] + b_i[i]
  except KeyError:
    return alpha
alpha, b_i, b_u, e = gradient_descent(x_train, y_train, b_u, b_i, alpha, 15, 1)
pred_test = [predict_test(d, b_u, b_i, alpha) for d in data_test]
print "MSE", mean_squared_error(pred_test, y_test)