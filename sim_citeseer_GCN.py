from __future__ import print_function

from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

from keras_gcn.layers.graph import GraphConvolution
from keras_gcn.utils import *
from keras_gat.utils import *

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression

import time
import math

# Define parameters
DATASET = 'citeseer'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_IT = 200
PATIENCE = 5  # early stopping patience
epsilon = 0.

# Get data
A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test, y_1 = load_data_1(DATASET)
X = preprocess_features(X)
y = np.ones((y_1.shape[0],2))
y[y_1[:,3]==1,0]=0
y[:,1] = y[:,1]-y[:,0]

r_0 = 60/np.sum(y,axis=0)[0]
r_1 = 120/np.sum(y,axis=0)[1]
a_0 = math.log(1/r_0-1)
a_1 = math.log(1/r_1-1)-a_0
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, idx_test_0, idx_test_1 = get_splits_1(y,a_0,a_1)


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

################################
########## Epoch 0 #############
################################
X_in = Input(shape=(X.shape[1],))
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

wait = 0
preds = None
best_val_loss = 99999

# Fit
for itr in range(1, NB_IT+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])

    print("Iteration: {:04d}".format(itr),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Iteration {}: early stopping'.format(itr))
            break
        wait += 1

test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
print(evaluate_preds(preds, [y_test], [idx_test_0]))
print(evaluate_preds(preds, [y_test], [idx_test_1]))

y_est = np.random.binomial(size=preds.shape[0], n=1, p= preds[:,1])
y_est[idx_train+idx_val] = y[idx_train+idx_val,1]
r = np.zeros((y.shape[0],))
r[idx_train+idx_val]=1
r_1 = np.sum(y_train+y_val,axis=0)[1]/np.sum(y_est)
r_0 = np.sum(y_train+y_val,axis=0)[0]/np.sum(1-y_est)
weights = [1, r_0/r_1]
weight_loss = weighted_categorical_crossentropy(weights)

################################
########## Epoch > 1 ###########
################################

y_est_old = np.zeros(y_est.shape)

epoch = 0
threshold = 0.05

while (np.sum(np.abs(y_est - y_est_old))/np.shape(y_est)[0]>threshold and epoch<5):
    y_est_old = y_est
    X_in = Input(shape=(X.shape[1],))
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
    model = Model(inputs=[X_in]+G, outputs=Y)
    model.compile(loss=weight_loss, optimizer=Adam(lr=0.005))

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999

    # Fit
    for itr in range(1, NB_IT+1):
        t = time.time()
        model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
        preds = model.predict(graph, batch_size=A.shape[0])
        train_val_loss, train_val_acc = evaluate_preds_1(preds, [y_train, y_val],
                                                   [idx_train, idx_val], weights)
        print("Iteration: {:04d}".format(itr),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Iteration {}: early stopping'.format(itr))
                break
            wait += 1
    # Testing
    test_loss, test_acc = evaluate_preds_1(preds, [y_test], [idx_test],weights)
    print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
    print(evaluate_preds(preds, [y_test], [idx_test_0]))
    print(evaluate_preds(preds, [y_test], [idx_test_1]))

    #y_est = np.random.binomial(size=preds.shape[0], n=1, p= preds[:,1])
    y_est = np.round(preds[:,1])
    y_est[idx_train+idx_val] = y[idx_train+idx_val,1]
    r = np.zeros((y.shape[0],))
    r[idx_train+idx_val]=1
    r_1 = np.sum(y_train+y_val,axis=0)[1]/np.sum(y_est)
    r_0 = np.sum(y_train+y_val,axis=0)[0]/np.sum(1-y_est)
    weights = [1, r_0/r_1]
    weight_loss = weighted_categorical_crossentropy(weights)

    epoch += 1
    print("Epoch= {:.4f}".format(epoch),
          "Epsilon= {:.4f}".format(np.sum(np.abs(y_est - y_est_old))/np.shape(y_est)[0]))
