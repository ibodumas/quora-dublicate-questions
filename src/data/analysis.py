# --------------------------------------------------------------------------- #

# Author: Ibrahim Odumas
# All Rights Reserved
# Open-source, free to copy

# --------------------------------------------------------------------------- #

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# ---------- Plot Setting---------- #
COLOR = sns.color_palette()
FIGSIZE = (10, 6)
FONTSIZE = 20
ALPHA_PLT = 1
# --------------------------------- #

# TRAIN_DF = pd.read_csv(os.path.join(os.getcwd(), "..", "..", "quoradata", "train.csv"))
# TEST_DF = pd.read_csv(os.path.join(os.getcwd(), "..", "..", "quoradata", "test.csv"))

TRAIN_DF = pd.read_csv(os.path.join("quoradata", "train.csv"))
TEST_DF = pd.read_csv(os.path.join("quoradata", "test.csv"))


def generate_word_cloud(text):
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(text, fontsize=FONTSIZE)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # ------- Analyzing Train set -------------#
    # 1. get the a table of 0 and 1 in the class label
    train_cl_tb = TRAIN_DF["is_duplicate"].value_counts()
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
        [TRAIN_DF.shape[0], TEST_DF.shape[0]],
        alpha=ALPHA_PLT,
        color=COLOR[3],
    )
    plt.ylabel("Frequency", fontsize=FONTSIZE)
    plt.xlabel("Sample Size", fontsize=FONTSIZE)
    plt.show()
    # ------- Analyzing training set ----------- #

    # ------- Word Cloud ----------------------- #
    text1 = TRAIN_DF.question1[0]  # Que1 with is_duplicate = 0
    text2 = TRAIN_DF.question2[0]  # Que2 with is_duplicate = 0
    text3 = TRAIN_DF.question1[5]  # Que1 with is_duplicate = 1
    text4 = TRAIN_DF.question2[5]  # Que2 with is_duplicate = 1
    generate_word_cloud(text1)
    generate_word_cloud(text2)
    generate_word_cloud(text3)
    generate_word_cloud(text4)