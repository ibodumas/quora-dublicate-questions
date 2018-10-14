import utils
import datetime
from time import time
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Model as keras_model
from keras.optimizers import Adadelta
from keras.layers import Input as keras_input
from keras.layers import Embedding as keras_embedding
from keras.layers import LSTM as keras_LSTM
from keras.layers import Lambda as keras_Lambda


n_hidden = utils.N_HIDDEN_LAYERS
gradient_clipping_norm = utils.GRADIENT_CLIPPING_NORM
n_epoch = utils.N_EPOCH


def exponent_neg_manhattan_distance(left, right):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


left_input = keras_input(shape=(utils.MAX_SEQ_LEN,), dtype='int32')
right_input = keras_input(shape=(utils.MAX_SEQ_LEN,), dtype='int32')

embedding_layer = keras_embedding(len(utils.EMBEDDING_MATRIX), utils.EMBEDDING_DIM, weights=[utils.EMBEDDING_MATRIX],
                                  input_length=utils.MAX_SEQ_LEN, trainable=False)


# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = keras_LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = keras_Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                         output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = keras_model.Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = time()

malstm_trained = malstm.fit([utils.X_TRAIN['left'], utils.X_TRAIN['right']], utils.Y_TRAIN,
                            batch_size=utils.BATCH_SIZE, nb_epoch=utils.N_EPOCH,
                            validation_data=([utils.X_VAL['left'], utils.X_VAL['right']], utils.Y_VAL))

print("Training time finished.\n{} epochs in {}".
      format(n_epoch, datetime.timedelta(seconds=time() - training_start_time)))


if utils.PLOT_MODEL_LOSS:
    # plot model accuracy
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
