from __future__ import print_function

from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *
from keras import backend as K
from keras_gat import GraphAttention

from sklearn.decomposition import IncrementalPCA

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

#y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
r_0 = 700/np.sum(y,axis=0)[0]
r_1 = 300/np.sum(y,axis=0)[1]
#a_0 = 2.65
#a_1 = -2.04
a_0 = math.log(1/r_0-1)
a_1 = math.log(1/r_1-1)-a_0
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, idx_test_0, idx_test_1 = get_splits_1(y,a_0,a_1)

def weighted_categorical_crossentropy(weights):

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

#weights = np.array([1,r_0/r_1]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#weight_loss = weighted_categorical_crossentropy(weights)

# Normalize X
X /= X.sum(1).reshape(-1, 1)

ipca = IncrementalPCA(n_components=2, batch_size=3)
ipca.fit(X)
IncrementalPCA(batch_size=3, copy=True, n_components=2, whiten=False)
X_1 = ipca.transform(X)
[r_0, r_1] = weight_cal(y_train, y_val, X_1)
weights = np.array([1,r_0/r_1])
weight_loss = weighted_categorical_crossentropy(weights)

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

X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
#H = GraphConvolution(4, support, activation='relu', kernel_regularizer=l2(5e-4),name='g1')([H]+G)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
#Y = Dense(y.shape[1],activation='softmax')(H)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
#model.compile(loss=weight_loss, optimizer=Adam(lr=0.005))

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    #train_val_loss, train_val_acc = evaluate_preds_1(preds, [y_train, y_val],
    #                                               [idx_train, idx_val], weights)
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])

    print("Epoch: {:04d}".format(epoch),
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
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
#test_loss, test_acc = evaluate_preds_1(preds, [y_test], [idx_test],weights)
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
print(evaluate_preds(preds, [y_test], [idx_test_0]))
print(evaluate_preds(preds, [y_test], [idx_test_1]))

g1_out = model.get_layer("g1").output
model_g = Model(inputs=[X_in]+G, outputs=g1_out)
g1_output = model_g.predict(graph,batch_size=A.shape[0])
