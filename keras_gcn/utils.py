from __future__ import print_function

import os
import pickle as pkl
import sys
import networkx as nx
import scipy.sparse as sp
import math
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from random import sample
from sklearn.metrics import mean_squared_error

from keras import backend as K


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


#def load_data(path="data/cora/", dataset="cora"):
def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

def get_splits_1(y,a_0,a_1):
    idx = np.arange(0,y.shape[0],1)
    r_0 = 1/(1+math.exp(a_0))
    r_1 = 1/(1+math.exp(a_0+a_1))
    tmp0 = y[:,0]==1
    tmp1 = y[:,1]==1
    idx_y0 = [i for i, x in enumerate(tmp0) if x]
    idx_y1 = [i for i, x in enumerate(tmp1) if x]
    #idx_y0 = idx[y[:,0]==1]
    #idx_y1 = idx[y[:,1]==1]
    idx_all_0 = sample(idx_y0,int(r_0*len(idx_y0)))
    idx_all_1 = sample(idx_y1,int(r_1*len(idx_y1)))
    idx_train_0 = sample(idx_all_0,int(len(idx_all_0)/2))
    idx_train_1 = sample(idx_all_1,int(len(idx_all_1)/2))
    idx_train = idx_train_0 + idx_train_1
    idx_val_0 = list(set(idx_all_0) - set(idx_train_0))
    idx_val_1 = list(set(idx_all_1) - set(idx_train_1))
    idx_val = idx_val_0 + idx_val_1
    #idx_val = range(200, 500)
    idx_test_0 = list(set(idx_y0) - set(idx_all_0))
    idx_test_1 = list(set(idx_y1) - set(idx_all_1))
    #idx_test = list(set(idx_y0+idx_y1) - set(idx_all_0+idx_all_1))
    idx_test = idx_test_0 + idx_test_1
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, idx_test_0, idx_test_1

def get_splits_2(y,r_1):
    tmp0 = y[:,0]==1
    tmp1 = y[:,1]==1
    idx_y0 = [i for i, x in enumerate(tmp0) if x]
    idx_y1 = [i for i, x in enumerate(tmp1) if x]
    idx_all_1 = [i for i, x in enumerate((r_1==1)*(y[:,1]==1)) if x]
    idx_all_0 = [i for i, x in enumerate((r_1==1)*(y[:,1]==0)) if x]
    idx_train_1 = sample(idx_all_1,int(len(idx_all_1)/2))
    idx_train_0 = sample(idx_all_0,int(len(idx_all_0)/2))
    idx_train = idx_train_0 + idx_train_1
    idx_all = idx_all_0 + idx_all_1
    #idx_train = sample(idx_all,int(len(idx_all)/2))
    idx_val = list(set(idx_all) - set(idx_train))
    idx_test_0 = list(set(idx_y0) - set(idx_all))
    idx_test_1 = list(set(idx_y1) - set(idx_all))
    idx_test = idx_test_0 + idx_test_1
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])

    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, idx_test_0, idx_test_1


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def weight_categorical_crossentropy(preds, labels, weights):
    weight = np.sum(labels*weights,axis=1)
    return np.mean(weight*(-np.log(np.extract(labels, preds))))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds_1(preds, labels, indices, weights):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        #split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_loss.append(weight_categorical_crossentropy(preds[idx_split], y_split[idx_split],weights))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

def evaluate_preds_2(preds, pi, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        tmp = y_split * pi
        split_loss.append(categorical_crossentropy(preds[idx_split], tmp[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def evaluate_preds_3(preds, labels, indices):

    split_mape = list()
    split_mse = list()

    for y_split, idx_split in zip(labels, indices):
        split_mse.append(mean_squared_error(preds[idx_split], y_split[idx_split]))
        split_mape.append(np.mean(np.abs((y_split[idx_split]-preds[idx_split])/y_split[idx_split])))

    return split_mse, split_mape

def evaluate_preds_4(preds, pi, labels, indices):

    split_mape = list()
    split_mse = list()

    for y_split, idx_split in zip(labels, indices):
        tmp0 = np.sqrt(pi)*preds
        tmp1 = np.sqrt(pi)*y_split
        split_mse.append(mean_squared_error(tmp0[idx_split], tmp1[idx_split]))
        split_mape.append(np.mean(np.abs((y_split[idx_split]-preds[idx_split])/y_split[idx_split])))

    return split_mse, split_mape


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def weight_cal(y_train, y_val, X_1):
    y = y_train + y_val
    tmp = X_1
    tmp[:,1] = tmp[:,0]
    [n_0,n_1] = np.sum(y,axis=0)
    [r_0,r_1] = np.sum(y*tmp,axis=0)
    n = y_train.shape[0]
    r = np.sum(tmp[:,0])
    p_1 = (r_1*n_0-r_0*n_1)/(r*n_0-r_0*n)
    p_0 = (r_1*n_0-r_0*n_1)/(r_1*n-r*n_1)

    return p_0, p_1

def weight_cal_1(y_train, y_val, X):
    y = y_train + y_val
    #tmp = np.zeros(y.shape)
    #tmp[:,0] = np.sum(X,axis=1)
    #tmp[:,1] = np.sum(X,axis=1)
    tmp1 = np.sum(X,axis=1)
    tmp = np.array(np.concatenate((tmp1,tmp1),axis=1))
    [n_0,n_1] = np.sum(y,axis=0)
    [r_0,r_1] = np.sum(y*tmp,axis=0)
    n = y_train.shape[0]
    r = np.sum(tmp[:,0])
    p_1 = (r_1*n_0-r_0*n_1)/(r*n_0-r_0*n)
    p_0 = (r_1*n_0-r_0*n_1)/(r_1*n-r*n_1)

    return p_0, p_1


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

def categorical_crossentropy_1(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred)
    loss = -K.sum(loss, -1)

    return loss

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
