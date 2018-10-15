###
import utils
from preproc import preprocessing
import datetime
from time import time
import os
import pandas as pd
from keras.models import Model as keras_model
from keras.optimizers import Adadelta
from keras.layers import Input as keras_input
from keras.layers import Embedding as keras_embedding
from keras.layers import LSTM as keras_LSTM
from keras.layers import Lambda as keras_Lambda
from keras.callbacks import ModelCheckpoint

TRAIN_DF = pd.read_csv(os.path.join(utils.ROOT_DIR, "quoradata", "train_.csv"))
X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, EMBEDDING_MATRIX = preprocessing(
    data_df=TRAIN_DF, is_training=True
)


left_input = keras_input(shape=(utils.MAX_SEQ_LEN,), dtype="int32")
right_input = keras_input(shape=(utils.MAX_SEQ_LEN,), dtype="int32")

embedding_layer = keras_embedding(
    len(EMBEDDING_MATRIX),
    utils.EMBEDDING_DIM,
    weights=[EMBEDDING_MATRIX],
    input_length=utils.MAX_SEQ_LEN,
    trainable=False,
)

encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)
shared_lstm = keras_LSTM(utils.N_HIDDEN_LAYERS)
left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)
malstm_distance = keras_Lambda(
    function=lambda x: utils.exponent_neg_manhattan_distance(x[0], x[1]),
    output_shape=lambda x: (x[0][0], 1),
)([left_output, right_output])

malstm = keras_model([left_input, right_input], [malstm_distance])


optimizer = Adadelta(clipnorm=utils.GRADIENT_CLIPPING_NORM)
malstm.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

start_time = time()

check_pointer = None
if utils.SAVE_MODEL:
    check_pointer = ModelCheckpoint(
        filepath=utils.DIR_WEIGHT_SAVING, verbose=2, save_best_only=True
    )

malstm.fit(
    [X_TRAIN["left"], X_TRAIN["right"]],
    Y_TRAIN,
    batch_size=utils.BATCH_SIZE,
    nb_epoch=utils.N_EPOCH,
    validation_data=([X_VAL["left"], X_VAL["right"]], Y_VAL),
    callbacks=[check_pointer],
)

print(
    "Training time finished.\n{} epochs in {}".format(
        utils.N_EPOCH, datetime.timedelta(seconds=time() - start_time)
    )
)
