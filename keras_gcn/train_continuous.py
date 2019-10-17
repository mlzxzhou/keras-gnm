from __future__ import print_function

from keras.layers import Input, Dropout, Dense, Add, concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *
from keras import backend as K

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
#import statsmodels.api as sm

import time
import math

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200  # early stopping patience

# Get data
X, A, y_1 = load_data(dataset=DATASET)
y = np.ones((y_1.shape[0],2))
y[y_1[:,3]==1,0]=0
y[:,1] = y[:,1]-y[:,0]

#######################################
######### Simulate data ###############
#######################################

g1_output = np.load('/kegra/g1_output.npy')
#beta = np.random.uniform(low=0.5, high=4,size=4).reshape(-1,1)
#beta = np.array([0.5,1,2,3]).reshape(-1,1)
beta_1 = 3
beta_0 = -3
#y = 2 + np.dot(g1_output,beta)+np.random.normal(0,0.5,g1_output.shape[0]).reshape(-1,1)
pi_y = 1/(1+np.exp(beta_0+beta_1*g1_output))
y = np.random.binomial(size= pi_y.shape[0], n=1, p=pi_y[:,0])
np.sum(y)


print(np.max(y),np.min(y))

pi = 1/(1+np.exp(1.5+2*y))
r_1 = np.random.binomial(size= pi.shape[0], n=1, p=pi)
np.sum(r_1)

model = LogisticRegression(fit_intercept = True)
mdl = model.fit(y, r_1)
print(mdl.coef_)
r_prob = mdl.predict_proba(y)

tmp0 = r_1==1
tmp1 = r_1==0
#tmp0 = r_1==0
#tmp1 = r_1==1
idx_all = [i for i, x in enumerate(tmp0) if x]
idx_train = sample(idx_all,int(len(idx_all)/2))
idx_val = list(set(idx_all) - set(idx_train))
idx_test = [i for i, x in enumerate(tmp1) if x]
y_train = np.zeros(y.shape)
y_val = np.zeros(y.shape)
y_test = np.zeros(y.shape)
y_train[idx_train] = y[idx_train]
y_val[idx_val] = y[idx_val]
y_test[idx_test] = y[idx_test]
train_mask = sample_mask(idx_train, y.shape[0])

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X]+T_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

#### Epoch 0 ####
X_in = Input(shape=(X.shape[1],))
#H = Dropout(0.5)(X_in)
#H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
#H = Dropout(0.5)(H)
#H = GraphConvolution(4, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
#Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in]+G)
H = GraphConvolution(1, support, activation='tanh', kernel_regularizer=l2(5e-4),name='g1')([H]+G)
Y = Dense(y.shape[1],activation='sotmax')(H)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
#model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1))
#model.compile(loss=weight_loss, optimizer=Adam(lr=0.005))

# Helper variables for main training
NB_EPOCH = 200
wait = 0
preds_y = None
PATIENCE = 20
best_val_mse = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds_y = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    #train_val_loss, train_val_acc = evaluate_preds_1(preds, [y_train, y_val],
    #                                               [idx_train, idx_val], weights)
    train_val_mse, train_val_mape = evaluate_preds_3(preds_y, [y_train, y_val],
                                                   [idx_train, idx_val])

    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_mse[0]),
          "val_loss= {:.4f}".format(train_val_mse[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_mse[1] < best_val_mse:
        best_val_mse = train_val_mse[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
#test_loss, test_acc = evaluate_preds_1(preds, [y_test], [idx_test],weights)
test_mse, test_mape = evaluate_preds_3(preds_y, [y_test], [idx_test])
print("Test set results:",
      "MSE= {:.4f}".format(np.sqrt(test_mse[0])),
      "MAPE= {:.4f}".format(test_mape[0]))

for layer in model.layers:
    weights = layer.get_weights()

#print(np.abs(beta-weights[0]),np.abs(2-weights[1]))


### weighted regression ###
pi_est = 1/r_prob[:,1]

X_in = Input(shape=(X.shape[1],))
#H = Dropout(0.5)(X_in)
#H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
#H = Dropout(0.5)(H)
#H = GraphConvolution(4, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
#Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in]+G)
H = GraphConvolution(1, support, activation='sigmoid', kernel_regularizer=l2(5e-4),name='g1')([H]+G)
Y = Dense(y.shape[1],activation='linear')(H)
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.02))

wait = 0
preds_y = None
best_val_mse = 999999
PATIENCE = 15
#NB_EPOCH

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask*pi_est,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds_y = model.predict(graph, batch_size=A.shape[0])

    train_val_mse, train_val_mape = evaluate_preds_4(preds_y, pi_est, [y_train, y_val],
                                                   [idx_train, idx_val])

    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_mse[0]),
          "val_loss= {:.4f}".format(train_val_mse[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_mse[1] < best_val_mse:
        best_val_mse = train_val_mse[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

test_mse, test_mape = evaluate_preds_3(preds_y, [y_test], [idx_test])
print("Test set results:",
      "MSE= {:.4f}".format(np.sqrt(test_mse[0])),
      "MAPE= {:.4f}".format(test_mape[0]))

for layer in model.layers:
    weights_1 = layer.get_weights()

print(np.abs(beta-weights[0]),np.abs(2-weights[1]))
print(np.abs(beta - weights_1[0]), np.abs(2 - weights_1[1]))
