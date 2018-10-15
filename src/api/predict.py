import utils
from preproc import preprocessing
import logging.config
from keras.models import model_from_json
from keras.optimizers import Adadelta
import pandas as pd


LOG = logging.getLogger(utils.NAME)

def predictor(question1, question2):
    data = pd.DataFrame({'question1': question1, "question2": question2}, index=[0])
    data = preprocessing(data, is_training=False)

    json_file = open(utils.DIR_MODEL_SAVING, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(utils.DIR_WEIGHT_SAVING)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    optimizer = Adadelta(clipnorm=utils.GRADIENT_CLIPPING_NORM)
    loaded_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    score = loaded_model.predict(data, batch_size=utils.BATCH_SIZE, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


















