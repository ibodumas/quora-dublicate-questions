###
import utils
import re
import itertools
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from keras.preprocessing.sequence import pad_sequences


def preprocessing(data_df, is_training):
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
    word_to_vec = utils.WORD_2_VECTOR

    que_cols = ['question1', 'question2']


    ###
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

    stops = set(utils.STOPWORDS)
    data_df = iter_over_df(data_df, que_cols, text_wordlist, stops, word_to_vec, vocab, inverse_vocab)


    ### Build the embedding matrix
    def build_embed_matr(vocab, word_to_vec, embed_dim):
        embed_mat = 1 * np.random.randn(len(vocab) + 1, embed_dim)
        embed_mat[0] = 0  # to ignore padding
        for word, index in vocab.items():
            if word in word_to_vec.vocab:
                embed_mat[index] = word_to_vec.word_vec(word)

        return embed_mat


    ### Zero padding
    def zero_padding_train(*args, Y_train=None, max_seq_len):
        data = [*args]
        for dataset, side in itertools.product(data, ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_len)

        if is_training:
            assert X_train['left'].shape == X_train['right'].shape
            assert len(X_train['left']) == len(Y_train)

        return data


    # Split to train validation
    if is_training:
        EMBEDDING_MATRIX = build_embed_matr(vocab, word_to_vec, utils.EMBEDDING_DIM)
        validation_size = int(0.1 * data_df.shape[0])
        X = data_df[que_cols]
        Y = data_df['is_duplicate']
        X_train, X_val, Y_train, Y_val = sklearn_train_test_split(X, Y, test_size=validation_size)
        del X, Y, validation_size, data_df, que_cols
        X_train = {'left': X_train.question1, 'right': X_train.question2}
        X_val = {'left': X_val.question1, 'right': X_val.question2}
        Y_train = Y_train.values
        Y_val = Y_val.values
        X_TRAIN, X_VAL = zero_padding_train(X_train, X_val, Y_train=Y_train, max_seq_len=utils.MAX_SEQ_LEN)
        Y_TRAIN = Y_train
        Y_VAL = Y_val
        del X_train, X_val, Y_train, Y_val
        return X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, EMBEDDING_MATRIX
    else:
        data_df = {'left': data_df.question1, 'right': data_df.question2}
        return zero_padding_train(data_df, Y_train=None, max_seq_len = utils.MAX_SEQ_LEN)[0]












