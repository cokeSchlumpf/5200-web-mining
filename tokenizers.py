import re
import string

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

parser = English()
punctuations = string.punctuation
stop_words = STOP_WORDS


def meta():
    return {
        "name": "Simple Default Tokenizer"
    }


def tokenizer(sentence):
    # Simple Text Cleansing
    sentence = sentence.strip().lower()

    sentence = re.sub('&#([a-zA-Z0-9]+);', r' \1 ', sentence)

    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    mytokens = [word for word in mytokens if word not in punctuations and word not in stop_words]

    # return preprocessed list of tokens
    return mytokens
