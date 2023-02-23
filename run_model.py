import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error




def datagen(df, seq_len, batch_size, targetcol, kind):
    "As a generator to produce samples for Keras model"
    batch = []
    while True:
        # Pick one dataframe from the pool
        # print('a')
        input_cols = [c for c in df.columns if c != targetcol]
        index = df.index[df.index < TRAIN_TEST_CUTOFF]
        split = int(len(index) * TRAIN_VALID_RATIO)
        if kind == 'train':
            index = index[:split]   # range for the training set
        elif kind == 'valid':
            index = index[split:]   # range for the validation set
        # Pick one position, then clip a sequence length
        while True:
            t = random.choice(index)      # pick one time step
            n = (df.index == t).argmax()  # find its position in the dataframe
            if n-seq_len+1 < 0:
                continue # can't get enough data for one sequence length
            frame = df.iloc[n-seq_len+1:n+1]
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            break
        # if we get enough for a batch, dispatch
        if len(batch) == batch_size:
        	X, y = zip(*batch)
        	X, y = np.expand_dims(np.array(X), 3), np.array(y)
        	yield X, y
        	batch = []


def cnnpred_2d(seq_len=60, n_features=4, n_filters=(8,8,8), droprate=0.1):
    "2D-CNNpred model according to the paper"
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(3, 1), activation="relu"),
        # MaxPool2D(pool_size=(2,1)),
        # Dropout(droprate),
        Conv2D(n_filters[1], kernel_size=(3,1), activation="relu"),
        Conv2D(n_filters[2], kernel_size=(3,1), activation="relu"),
        # MaxPool2D(pool_size=2),
        Flatten(),
        # Dropout(droprate),
        # Dense(19, activation="sigmoid"),
        Dense(8, activation="sigmoid"),
        # Dropout(droprate),
        Dense(1, activation="sigmoid")
    ])
    return model

def recall_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1macro(y_true, y_pred):
	f_pos = f1_m(y_true, y_pred)
	# negative version of the data and prediction
	f_neg = f1_m(1-y_true, 1-K.clip(y_pred,0,1))
	return (f_pos + f_neg)/2

def hit_rate(y_true, y_pred):
    x = 0
    for i in range(len(y_pred)): 
        x = x + y_pred[i][0]*y_true[i]
    y = y_pred.sum()
    return (x / y) if y else 0 


def testgen(df, seq_len, targetcol):
    "Return array of all test samples"
    batch = []
    input_cols = [c for c in df.columns if c != targetcol]
    # find the start of test sample
    t = df.index[df.index >= TRAIN_TEST_CUTOFF][0]
    n = (df.index == t).argmax()
    # extract sample using a sliding window
    for i in range(n+1, len(df)+1):
        frame = df.iloc[i-seq_len:i]
        batch.append([frame[input_cols].values, frame[targetcol][-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X),3), np.array(y)

seq_len    = 60
batch_size = 128
n_epochs   = 3
n_features = 4

########################################################################################
####################### Execute using below script #####################################


seq_len    = 60
batch_size = 128
n_epochs   = 12
n_features = 4

class_weight = {0: 1., 1: 3.}
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
model = cnnpred_2d(seq_len, n_features)
model.compile(optimizer="adam", loss="mae", metrics=["accuracy", precision_m])
model.summary()  # print model structure to console
checkpoint_path = "./cp2d-{epoch}-{val_precision_m}.h5"
callbacks = [
ModelCheckpoint(checkpoint_path,
    monitor='val_accuracy',
    #monitor = ["acc", f1macro], 
    mode="max",
    verbose=0, save_best_only=True, save_weights_only=False, save_freq="epoch")
]

K.set_value(model.optimizer.learning_rate, 0.0001)



model.fit(
    datagen(x, seq_len, batch_size, "Target", "train"),
    validation_data=datagen(x, seq_len, batch_size, "Target", "valid"),
    epochs=n_epochs, steps_per_epoch=400, validation_steps=10, verbose=1,
    callbacks=callbacks, class_weight=class_weight
          )


# Prepare test data
test_data, test_target = testgen(x, seq_len, "Target")
 
# Test the model
test_out = model.predict(test_data)
test_pred = (test_out > 0.5).astype(int)
print("accuracy:", accuracy_score(test_pred, test_target))
print("MAE:", mean_absolute_error(test_pred, test_target))
print("F1:", f1_score(test_pred, test_target))


for i in range(len(test_pred)):
    print(test_pred[i], test_target[i], test_out[i])

hit_rate(test_target, test_pred)





















########################################################################################
######################### Iterate script ###############################################

# parameters to vary: # of epochs, weights, win threshold
# output: hit rate, win threshold, 
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

n_epochs   = 15
class_weight = {0: 1., 1: 3.}
K.set_value(model.optimizer.learning_rate, 0.001)
model = cnnpred_2d(seq_len, n_features)
model.compile(optimizer="adam", loss="mae", metrics=["accuracy", precision_m])
model.summary()  # print model structure to console
checkpoint_path = "./cp2d-{epoch}-{val_precision_m}.h5"
callbacks = [
ModelCheckpoint(checkpoint_path,
    monitor='val_accuracy',
    #monitor = ["acc", f1macro], 
    mode="max",
    verbose=0, save_best_only=True, save_weights_only=False, save_freq="epoch")
]
meta_data_df = pd.DataFrame(index=range(9),columns=range(5))
meta_data_list = [0.0]*5
model.fit(
        datagen(x, seq_len, batch_size, "Target", "train"),
        validation_data=datagen(x, seq_len, batch_size, "Target", "valid"),
        epochs=n_epochs, steps_per_epoch=400, validation_steps=10, verbose=1,
        callbacks=callbacks, class_weight=class_weight
    )
test_data, test_target = testgen(x, seq_len, "Target")
# Test the model
test_out = model.predict(test_data)
for j in np.linspace(.5,.9,5):
    test_pred = (test_out > j).astype(int)
    hit_pct = hit_rate(test_target, test_pred)
    meta_data_list[((j*10)-5).astype(int)] = hit_pct

########################################################################################

# To Do
# 1. add aditional stocks (3rd dimenstion)
# 2. compute the correct class weights
# 3. test alternative architectures
# 4. build type precision calculators
# 5. build iterator and output metrics to file



# meta_data_df = pd.DataFrame(index=range(9),columns=range(5))
#
# for i in range(1,10):
#     # returns: hit rate for various class weights and probabilities
# class_weight[0] = 1
# class_weight[1] = i
# model.fit(
# datagen(x, seq_len, batch_size, "Target", "train"),
# validation_data=datagen(x, seq_len, batch_size, "Target", "valid"),
# epochs=n_epochs, steps_per_epoch=400, validation_steps=10, verbose=1,
# callbacks=callbacks, class_weight=class_weight
# )
# test_data, test_target = testgen(x, seq_len, "Target")
# # Test the model
# test_out = model.predict(test_data)
# # for j in np.linspace(.5,.9,5):
# #     print(j)
# #     test_pred = (test_out > j).astype(int)
# #     hit_pct = hit_rate(test_target, test_pred)
# #     print(hit_pct)
# #     meta_data_df.at[i-1, (j*10)-5] = hit_pct







with pd.ExcelWriter('output.xlsx') as writer:  
    meta_data_df.to_excel(writer, sheet_name='Sheet_name_1')











