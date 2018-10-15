##
import utils
from preproc import preprocessing
import pandas as pd
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras import backend as K


def predictor(question1, question2):
    data = pd.DataFrame({"question1": question1, "question2": question2}, index=[0])
    data = preprocessing(data, is_training=False)

    with CustomObjectScope(
        {"exponent_neg_manhattan_distance": utils.exponent_neg_manhattan_distance}
    ):
        model = load_model(utils.DIR_WEIGHT_SAVING)

    score = model.predict([data["left"], data["right"]])
    prob = score[0][0]
    is_duplicate = False
    if prob >= 0.5:
        is_duplicate = True

    del model, data, score
    K.clear_session()
    return {"is_duplicate": is_duplicate, "probability": str(prob)}
