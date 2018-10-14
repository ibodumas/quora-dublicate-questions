import os
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_MODEL_SAVING = os.path.join(ROOT_DIR, "quoradata", "model")
Google_Word_Vectors = os.path.join(ROOT_DIR, "quoradata", "GoogleNews-vectors-negative300.bin.gz")

nltk.download("stopwords")
STOPWORDS = stopwords.words('english')
TRAIN_DF = pd.read_csv(os.path.join(ROOT_DIR, "quoradata", "train.csv"))
# test_df = pd.read_csv(os.path.join(utils.ROOT_DIR, "quoradata", "test.csv"))

PLOT_MODEL_LOSS = True
MAX_SEQ_LEN = 250
EMBEDDING_MATRIX = None
EMBEDDING_DIM = 300
N_HIDDEN_LAYERS = 50
GRADIENT_CLIPPING_NORM = 1.25
BATCH_SIZE = 64
N_EPOCH = 25

X_TRAIN = None
Y_TRAIN = None
X_VAL = None
Y_VAL = None


if __name__ == "__main__":
    _path = [
                os.path.join(ROOT_DIR, "src"),
                os.path.join(ROOT_DIR, "src", "data"),
                os.path.join(ROOT_DIR, "src", "api"),
                os.path.join(ROOT_DIR, "src", "evaluate"),
                os.path.join(ROOT_DIR, "src", "model"),
                os.path.join(ROOT_DIR, "src", "quoradata")
            ]

    for pth in _path:
        if pth not in sys.path:
            sys.path.append(pth)


