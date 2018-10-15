# --------------------------------------------------------------------------- #

# Author: Ibrahim Odumas
# All Rights Reserved
# Open-source, free to copy

# --------------------------------------------------------------------------- #
import utils
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# ---------- Plot Setting ---------- #
COLOR = sns.color_palette()
FIGSIZE = (10, 6)
FONTSIZE = 20
ALPHA_PLT = 1
# ---------------------------------- #

# ---------- Load Data ------------- #
_TRAIN = TRAIN_DF = pd.read_csv(os.path.join(utils.ROOT_DIR, "quoradata", "train.csv"))
_TEST = pd.read_csv(os.path.join(utils.ROOT_DIR, "quoradata", "test.csv"))
columns = ['question1', 'question2']
TRAIN_X = _TRAIN[columns]
TRAIN_Y = _TRAIN.is_duplicate
TEST_X = _TEST[columns]
del _TRAIN, _TEST
# ---------------------------------- #


def generate_word_cloud(text):
    wc = WordCloud(background_color="white", max_words=20, stopwords=stopwords,
                   contour_width=3, contour_color='steelblue')
    wc.generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.title(text, fontsize=FONTSIZE)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # ---------- STOPWORDS ------------- #
    stopwords = set(STOPWORDS)
    stopwords.add("said")
    # ---------------------------------- #

    # ------- Analyzing Train set ------ #
    # 1. get the a table of 0 and 1 in the class label
    train_cl_tb = TRAIN_Y.value_counts()
    print(tabulate(pd.DataFrame(train_cl_tb), tablefmt="psql", headers="keys"))
    plt.figure(figsize=FIGSIZE)
    sns.barplot(train_cl_tb.index, train_cl_tb.values, alpha=ALPHA_PLT, color=COLOR[3])
    plt.ylabel("Frequency", fontsize=FONTSIZE)
    plt.xlabel("Is Duplicate - (Response Variable)", fontsize=FONTSIZE)
    plt.show()

    # 2. sample size of Train vs. Test
    plt.figure(figsize=FIGSIZE)
    sns.barplot(
        ["Train", "Test"],
        [TRAIN_X.shape[0], TEST_X.shape[0]],
        alpha=ALPHA_PLT,
        color=COLOR[3],
    )
    plt.ylabel("Frequency", fontsize=FONTSIZE)
    plt.xlabel("Sample Size", fontsize=FONTSIZE)
    plt.show()
    # ------ END Analyzing training set ---- #

    # ------- Generate Word Cloud ----------------------- #
    text1 = TRAIN_X.question1[0]  # Que1 with is_duplicate = 0
    text2 = TRAIN_X.question2[0]  # Que2 with is_duplicate = 0
    text3 = TRAIN_X.question1[5]  # Que1 with is_duplicate = 1
    text4 = TRAIN_X.question2[5]  # Que2 with is_duplicate = 1
    generate_word_cloud(text1)
    generate_word_cloud(text2)
    generate_word_cloud(text3)
    generate_word_cloud(text4)
    # ------- END Generate Word Cloud ----------------------- #
