import os
import sys
import nltk
from nltk.corpus import stopwords
import gensim.models as gensim_models
import keras.backend as K

NAME = "Natural Language Processing"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print("Loading Google_Word_Vectors Again")
Google_Word_Vectors = os.path.join(
    ROOT_DIR, "quoradata", "GoogleNews-vectors-negative300.bin.gz"
)
WORD_2_VECTOR = gensim_models.KeyedVectors.load_word2vec_format(
    Google_Word_Vectors, binary=True
)

SAVE_MODEL = True
DIR_WEIGHT_SAVING = os.path.join(ROOT_DIR, "quoradata", "model_json.h5")

nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
# test_df = pd.read_csv(os.path.join(utils.ROOT_DIR, "quoradata", "test.csv"))

PLOT_MODEL_LOSS = False
MAX_SEQ_LEN = 300
EMBEDDING_DIM = 300
N_HIDDEN_LAYERS = 50
GRADIENT_CLIPPING_NORM = 1.25
BATCH_SIZE = 64
N_EPOCH = 1  # 25


# API SETTINGS
CONNEX_SWAGGER = "api_spec_model.yml"
SWAGGER_UI = True
DEBUG = True
STANDARD_ERRORS = [501, 503, 400, 401, 403, 404, 405, 406, 409, 410, 412, 422, 428]


def exponent_neg_manhattan_distance(left, right):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


if __name__ == "__main__":
    _path = [
        os.path.join(ROOT_DIR, "src"),
        os.path.join(ROOT_DIR, "src", "learning"),
        os.path.join(ROOT_DIR, "src", "api"),
        os.path.join(ROOT_DIR, "src", "quoradata"),
    ]

    for pth in _path:
        if pth not in sys.path:
            sys.path.append(pth)
