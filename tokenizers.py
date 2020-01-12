import re
import string

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

parser = English()
punctuations = string.punctuation
stop_words = STOP_WORDS

REPLACE_NAMES = re.compile(r"@[a-zA-Z0-9_:]+")
REPLACE_EMOJIS = re.compile(r"&#\d+;")
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]/]")
REPLACE_URL = re.compile('http.?://[^\s]+[\s]?')


def meta():
    return {
        "name": "Simple Default Tokenizer"
    }


def tokenizer(sentence):
    # Simple Text Cleansing
    sentence = sentence.strip().lower()

    sentence = re.sub('&#([a-zA-Z0-9]+);', r' EMOJI_\1 ', sentence)
    sentence = re.sub(r'(@[a-zA-Z0-9_:]+)', r' TALK_TO_SOMEONES ', sentence)

    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    # mytokens = [word for word in mytokens if word not in punctuations and word not in stop_words]
    mytokens = [word for word in mytokens if word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens
