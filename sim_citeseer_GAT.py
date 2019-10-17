from __future__ import print_function

from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

from keras_gat import GraphAttention
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
A_1 = A + np.eye(A.shape[0])  # Add self-loops

N = X.shape[0]                # Number of nodes in the graph
F = X.shape[1]                # Original feature dimension
F_ = 8                        # Output size of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 5e-3          # Learning rate for Adam
es_patience = 100

r_0 = 60/np.sum(y,axis=0)[0]
r_1 = 120/np.sum(y,axis=0)[1]
a_0 = math.log(1/r_0-1)
a_1 = math.log(1/r_1-1)-a_0
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, idx_test_0, idx_test_1 = get_splits_1(y,a_0,a_1)

################################
########## Epoch 0 #############
################################

# Model definition (as per Section 3.3 of the paper)
X_in = Input(shape=(X.shape[1],))
A_in = Input(shape=(A_1.shape[0],))
dropout1 = Dropout(0.5)(X_in)
graph_attention_1 = GraphAttention(16,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])
dropout2 = Dropout(0.5)(graph_attention_1)
Y = GraphAttention(y.shape[1],
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])
model = Model(inputs=[X_in, A_in], outputs=Y)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
wait = 0
preds = None
best_val_loss = 99999

# Fit
for itr in range(1, NB_IT+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit([X, A_1],
          y_train,
          sample_weight=train_mask,
          epochs=1,
          batch_size=A_1.shape[0],
          #validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          #callbacks=[es_callback, tb_callback, mc_callback]
          )

    # Predict on full dataset
    preds = model.predict([X, A_1], batch_size=A_1.shape[0])

    # Train / validation scores
    #train_val_loss, train_val_acc = evaluate_preds_1(preds, [y_train, y_val],
    #                                               [idx_train, idx_val], weights)
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
    A_in = Input(shape=(A_1.shape[0],))
    dropout1 = Dropout(0.5)(X_in)
    graph_attention_1 = GraphAttention(16,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])
    dropout2 = Dropout(0.5)(graph_attention_1)
    Y = GraphAttention(y.shape[1],
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])
    model = Model(inputs=[X_in, A_in], outputs=Y)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
              loss=weight_loss,
              weighted_metrics=['acc'])

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999

    # Fit
    for itr in range(1, NB_IT+1):
        t = time.time()
        model.fit([X, A_1],
          y_train,
          sample_weight=train_mask,
          epochs=1,
          batch_size=A_1.shape[0],
          #validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          #callbacks=[es_callback, tb_callback, mc_callback]
          )
        preds = model.predict([X, A_1], batch_size=A_1.shape[0])
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
