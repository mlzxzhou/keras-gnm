n_attn_heads = 8
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 5e-3          # Learning rate for Adam
epochs = 10000                # Number of training epochs
es_patience = 100

A_1 = A + np.eye(A.shape[0])

X_in = Input(shape=(X.shape[1],))
A_in = Input(shape=(A_1.shape[0],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
dropout1 = Dropout(0.5)(X_in)
graph_attention_1 = GraphAttention(16,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])
dropout2 = Dropout(0.5)(graph_attention_1)
graph_attention_2 = GraphAttention(16,
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])
#Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
Y = Dense(y.shape[1],activation='softmax')(graph_attention_2)

# Compile model
model = Model(inputs=[X_in, A_in], outputs=Y)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

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


    #model.fit(graph, y_train, sample_weight=train_mask,
    #          batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict([X, A_1], batch_size=A_1.shape[0])

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