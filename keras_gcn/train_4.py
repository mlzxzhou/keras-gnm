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

import time
import math

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience

# Get data
X, A, y_1 = load_data(dataset=DATASET)
y = np.ones((y_1.shape[0],2))
y[y_1[:,3]==1,0]=0
y[:,1] = y[:,1]-y[:,0]

### Simulate g_2(x) ###
tmp = np.sum(X,axis=1)  ## \sum_p x_p
g2 = np.array(np.exp(tmp/13-4)-(tmp-15)/15)
pi = 1/(1+30*np.exp(g2[:,0]-1.7*y[:,1]))
r_1 = np.random.binomial(size= pi.shape[0], n=1, p=pi)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, idx_test_0, idx_test_1 = get_splits_2(y,r_1)
print(np.sum(y_train,axis=0))
print(np.sum(y_val,axis=0))

#weights = np.array([1,r_0/r_1]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#weight_loss = weighted_categorical_crossentropy(weights)

# Normalize X
#X /= X.sum(1).reshape(-1, 1)


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
r = np.zeros((y.shape))
r[idx_train+idx_val,1]=1
r[:,0] = 1 - r[:,1]

X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
#Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
Y = Dense(y.shape[1],activation='softmax')(H)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
#model.compile(loss=weight_loss, optimizer=Adam(lr=0.005))

# Helper variables for main training loop
wait = 0
preds_y = None
best_val_loss = 99999

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
    train_val_loss, train_val_acc = evaluate_preds(preds_y, [y_train, y_val],
                                                   [idx_train, idx_val])

    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

X_in_g2 = Input(shape=(X.shape[1],))
y_in = Input(shape=(1,))
g2 = Dense(128,activation='relu')(X_in_g2)
g2 = Dense(64,activation='relu')(g2)
g2 = Dense(32,activation='relu')(g2)
g2 = Dense(1,activation='tanh')(g2)
add = concatenate([g2, y_in])
r_out = Dense(2,activation='softmax')(add)

#y_est = np.ones((y.shape))
#y_est[:,1] = np.random.binomial(size=preds_y.shape[0], n=1, p=preds_y[:,1])
#y_est[idx_train+idx_val,1] = y[idx_train+idx_val,1]
y_est = np.ones((y.shape[0],1))

model_r = Model(inputs=[X_in_g2,y_in], outputs=r_out)
model_r.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005))

PATIENCE_0 = 10
wait = 0
best_val_acc = 0
preds_r = None
for epoch in range(1, NB_EPOCH+1):
    t = time.time()
    model_r.fit([X,y_est], r, epochs=1, shuffle=False, verbose=0)
    #model.fit(X, r, epochs=1, shuffle=False, verbose=0, batch_size=32)
    preds_r = model_r.predict([X,y_est])
    #preds = model.predict(X)
    train_val_loss = categorical_crossentropy(preds_r, r)
    train_val_acc = accuracy(preds_r, r)
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss),
          "train_acc= {:.4f}".format(train_val_acc),
          "time= {:.4f}".format(time.time() - t))

    if train_val_acc > best_val_acc:
        best_val_acc  = train_val_acc
        wait = 0
    else:
        if wait >= PATIENCE_0:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

pi_est = 1/preds_r
pi_est[:,0] = pi_est[:,1]

print(np.mean(pi_est[y[:,1]==1]),np.mean(pi_est[y[:,1]==0]),np.mean(1/pi[y[:,1]==1]),np.mean(1/pi[y[:,1]==0]))

#### Update P(y|g_1(x)) ####
X_in = Input(shape=(X.shape[1],))
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
Y = Dense(y.shape[1],activation='softmax')(H)

wait = 0
preds_y = None
best_val_loss_y = 99999

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
#model.compile(loss=weight_loss, optimizer=Adam(lr=0.005))

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train*pi_est, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds_y = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    #train_val_loss, train_val_acc = evaluate_preds_1(preds, [y_train, y_val],
    #                                               [idx_train, idx_val], weights)
    train_val_loss, train_val_acc = evaluate_preds_2(preds_y, pi_est, [y_train, y_val],
                                                   [idx_train, idx_val])

    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss_y :
        best_val_loss_y  = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
#test_loss, test_acc = evaluate_preds_1(preds, [y_test], [idx_test],weights)
test_loss, test_acc = evaluate_preds_2(preds_y, pi_est, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
print(evaluate_preds(preds_y, [y_test], [idx_test_0]))
print(evaluate_preds(preds_y, [y_test], [idx_test_1]))
    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
#test_loss, test_acc = evaluate_preds_1(preds, [y_test], [idx_test],weights)
test_loss, test_acc = evaluate_preds(preds_y, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
print(evaluate_preds(preds_y, [y_test], [idx_test_0]))
print(evaluate_preds(preds_y, [y_test], [idx_test_1]))


### Epoch > 1 ###
X_in_g2 = Input(shape=(X.shape[1],))
y_in = Input(shape=(1,))
g2 = Dense(128,activation='relu')(X_in_g2)
g2 = Dense(64,activation='relu')(g2)
g2 = Dense(32,activation='relu')(g2)
g2 = Dense(1,activation='tanh')(g2)
add = concatenate([g2, y_in])
r_out = Dense(2,activation='softmax')(add)

#y_est = np.ones((y.shape))
#y_est[:,1] = np.random.binomial(size=preds_y.shape[0], n=1, p=preds_y[:,1])
#y_est[idx_train+idx_val,1] = y[idx_train+idx_val,1]
y_est = np.ones((y.shape[0],1))

model_r = Model(inputs=[X_in_g2,y_in], outputs=r_out)
model_r.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005))

PATIENCE_0 = 10
wait = 0
best_val_acc = 0
preds_r = None
for epoch in range(1, NB_EPOCH+1):
    t = time.time()
    model_r.fit([X,y_est], r, epochs=1, shuffle=False, verbose=0)
    #model.fit(X, r, epochs=1, shuffle=False, verbose=0, batch_size=32)
    preds_r = model_r.predict([X,y_est])
    #preds = model.predict(X)
    train_val_loss = categorical_crossentropy(preds_r, r)
    train_val_acc = accuracy(preds_r, r)
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss),
          "train_acc= {:.4f}".format(train_val_acc),
          "time= {:.4f}".format(time.time() - t))

    if train_val_acc > best_val_acc:
        best_val_acc  = train_val_acc
        wait = 0
    else:
        if wait >= PATIENCE_0:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

pi_est = 1/preds_r
pi_est[:,0] = pi_est[:,1]

print(np.mean(pi_est[y[:,1]==1]),np.mean(pi_est[y[:,1]==0]),np.mean(1/pi[y[:,1]==1]),np.mean(1/pi[y[:,1]==0]))

#### Update P(y|g_1(x)) ####
X_in = Input(shape=(X.shape[1],))
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
Y = Dense(y.shape[1],activation='softmax')(H)

wait = 0
preds_y = None
best_val_loss_y = 99999

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
#model.compile(loss=weight_loss, optimizer=Adam(lr=0.005))

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train*pi_est, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds_y = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    #train_val_loss, train_val_acc = evaluate_preds_1(preds, [y_train, y_val],
    #                                               [idx_train, idx_val], weights)
    train_val_loss, train_val_acc = evaluate_preds_2(preds_y, pi_est, [y_train, y_val],
                                                   [idx_train, idx_val])

    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss_y :
        best_val_loss_y  = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
#test_loss, test_acc = evaluate_preds_1(preds, [y_test], [idx_test],weights)
test_loss, test_acc = evaluate_preds_2(preds_y, pi_est, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
print(evaluate_preds(preds_y, [y_test], [idx_test_0]))
print(evaluate_preds(preds_y, [y_test], [idx_test_1]))

