# --------------------------------------------------------------------------- #

# Author: Ibrahim Odumas
# All Rights Reserved
# Open-source, free to copy

# --------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


TRAIN_DF = pd.read_csv("../quoradata/train.csv")
TEST_DF = pd.read_csv("../quoradata/test.csv")
