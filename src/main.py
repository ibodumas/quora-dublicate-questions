###
import utils
from preproc import preprocessing
import datetime
from time import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Model as keras_model
from keras.optimizers import Adadelta
from keras.layers import Input as keras_input
from keras.layers import Embedding as keras_embedding
from keras.layers import LSTM as keras_LSTM
from keras.layers import Lambda as keras_Lambda


###
def exponent_neg_manhattan_distance(left, right):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


left_input = keras_input(shape=(utils.MAX_SEQ_LEN,), dtype='int32')
right_input = keras_input(shape=(utils.MAX_SEQ_LEN,), dtype='int32')

embedding_layer = keras_embedding(len(utils.EMBEDDING_MATRIX), utils.EMBEDDING_DIM, weights=[utils.EMBEDDING_MATRIX],
                                  input_length=utils.MAX_SEQ_LEN, trainable=False)


### Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)
del embedding_layer

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = keras_LSTM(utils.N_HIDDEN_LAYERS)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)
del encoded_left, encoded_right, shared_lstm

# Calculates the distance as defined by the MaLSTM model_json
malstm_distance = keras_Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                         output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

del left_output, right_output

### Pack it all up into a model_json
malstm = keras_model([left_input, right_input], [malstm_distance])
del left_input, right_input, malstm_distance

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=utils.GRADIENT_CLIPPING_NORM)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
del optimizer

### Preprocessing
TRAIN_DF = pd.read_csv(os.path.join(utils.ROOT_DIR, "quoradata", "train.csv"))
X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = preprocessing(TRAIN_DF, True)



# Start training
start_time = time()

###
malstm_trained = malstm.fit([X_TRAIN['left'], X_TRAIN['right']], Y_TRAIN,
                            batch_size=utils.BATCH_SIZE, nb_epoch=utils.N_EPOCH,
                            validation_data=([X_VAL['left'], X_VAL['right']], Y_VAL))
del malstm

# serialize model_json to JSON
model_json = malstm_trained.model.to_json()
with open(utils.DIR_MODEL_SAVING, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
malstm_trained.model.save_weights('/home/ib/techfield/pj/quora-dublicate-questions/quoradata/model_json.h5')


print("Training time finished.\n{} epochs in {}".
      format(utils.N_EPOCH, datetime.timedelta(seconds=time() - start_time)))




###
if utils.PLOT_MODEL_LOSS:
    # plot model_json accuracy
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
