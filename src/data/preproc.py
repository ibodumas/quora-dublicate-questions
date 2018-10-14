import utils
import re
import itertools
import gensim.models as gensim_models
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from keras.preprocessing.sequence import pad_sequences


train_df = utils.TRAIN_DF
stops = set(utils.STOPWORDS)


### Create embedding matrix
def text_wordlist(text):
    """
    Pre process and convert texts to a list of words
    :param text: text(sentence)
    :return: text
    """

    text = str(text)
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


### Prepare embedding
vocab = dict()  # word vocabulary
inverse_vocab = ['<unk>']
word_to_vec = gensim_models.KeyedVectors.load_word2vec_format(utils.Google_Word_Vectors, binary=True)

que_cols = ['question1', 'question2']


def iter_over_df(df, que_cols, text_wordlist, stops, word_to_vec, vocab, inverse_vocab):
    """
    Iterate over the questions
    :param df: dataframe
    :return: dataframe
    """
    for index, row in df.iterrows():
        for que in que_cols:

            que_to_vec = []
            for word in text_wordlist(row[que]):

                # skip unwanted words
                if word in stops and word not in word_to_vec.vocab:
                    continue

                if word not in vocab:
                    vocab[word] = len(inverse_vocab)
                    que_to_vec.append(len(inverse_vocab))
                    inverse_vocab.append(word)
                else:
                    que_to_vec.append(vocab[word])

            df.set_value(index, que, que_to_vec)

    return df


train_df = iter_over_df(train_df, que_cols, text_wordlist, stops, word_to_vec, vocab, inverse_vocab)


# Build the embedding matrix
def build_embed_matr(vocab, word_to_vec, embed_dim):
    embed_mat = 1 * np.random.randn(len(vocab) + 1, embed_dim)
    embed_mat[0] = 0  # to ignore padding
    for word, index in vocab.items():
        if word in word_to_vec.vocab:
            embed_mat[index] = word_to_vec.word_vec(word)

    return embed_mat


utils.EMBEDDING_MATRIX = build_embed_matr(vocab, word_to_vec, utils.EMBEDDING_DIM)
del stops, word_to_vec, vocab, inverse_vocab


# MAX_SEQ_LEN = max(train_df.question1.map(lambda x: len(x)).max(), train_df.question2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = int(0.1 * train_df.shape[0])
X = train_df[que_cols]
Y = train_df['is_duplicate']
X_train, X_val, Y_train, Y_val = sklearn_train_test_split(X, Y, test_size=validation_size)
del X, Y, validation_size, train_df


X_train = {'left': X_train.question1, 'right': X_train.question2}
X_val = {'left': X_val.question1, 'right': X_val.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_val = Y_val.values


# Zero padding
def zero_padding(X_train, X_validation, Y_train, max_seq_len):
    for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_len)

    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    return X_train, X_validation


utils.X_TRAIN, utils.X_VAL = zero_padding(X_train, X_val, Y_train, utils.MAX_SEQ_LEN)
utils.Y_TRAIN = Y_train
utils.Y_VAL = Y_val
del X_train, X_val, Y_train, Y_val





