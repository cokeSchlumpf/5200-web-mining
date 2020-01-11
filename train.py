from tokenizers import tokenizer as tokenizer_func, meta as tokenizer_meta

TRAINING_SET = './data/train.tsv'
TEST_SET = './data/test.tsv'

MODEL_PATH = './model/'
METRIC_PATH = './metrics/'


class DescribedComponent:

    meta = {}

    component = None

    def __init__(self, meta, component):
        self.meta = meta
        self.component = component


def create_classifier():
    # from sklearn.linear_model import LinearRegression
    # from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    params = {}

    classifier = LinearSVC(**params)

    meta = {
        "name": str(type(classifier)),
        "params": params
    }

    return DescribedComponent(meta, classifier)


def create_tokenizer():
    return DescribedComponent(tokenizer_meta(), tokenizer_func)


def create_vectorizer(tokenizer):
    from sklearn.feature_extraction.text import CountVectorizer

    params = {
        "ngram_range": (1, 1)
    }

    vectorizer = CountVectorizer(tokenizer=tokenizer, **params)

    meta = {
        "name": str(type(vectorizer)),
        "params": params
    }

    return DescribedComponent(meta, vectorizer)


def load_data(file):
    import pandas as pd
    return pd.read_csv(file, sep='\t', header=None, names=["hate", "offensive", "text"])


def calculate_metrics(model, X_test, y_test):
    from sklearn import metrics
    from scipy.stats import pearsonr

    predicted = model.predict(X_test)

    metrics = {
        "Accuracy": metrics.accuracy_score(y_test, predicted),
        "Precession": metrics.precision_score(y_test, predicted, average='micro'),
        "Recall": metrics.recall_score(y_test, predicted, average='micro'),
        "MeanAbsoluteError": metrics.mean_absolute_error(y_test, predicted),
        "PearsonCorrelation": pearsonr(y_test, predicted)[0]
    }

    return metrics


def train(field='hate'):
    import json
    import os
    from pathlib import Path

    from joblib import dump

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    # Get Vectorizer and Classifier
    tokenizer = create_tokenizer()
    vectorizer = create_vectorizer(tokenizer.component)
    classifier = create_classifier()

    # Build Pipeline
    pipe = Pipeline([
        ('vectorizer', vectorizer.component),
        ('classifier', classifier.component)])

    # Train Model
    print("... start model training")
    
    data = load_data(TRAINING_SET)
    X = data['text']  # the features we want to analyze
    y = data[field]  # the labels, or answers, we want to test against

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    pipe = pipe.fit(X_train, y_train)

    print("... done.")

    # Calculate Training Metrics
    print("... calculate metrics")

    metrics = calculate_metrics(pipe, X_test, y_test)
    meta = {
        "field": field,
        "pipeline": [tokenizer.meta, vectorizer.meta, classifier.meta],
        "training": metrics
    }

    print("... done.")

    # Save model
    print("... saving model")
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    Path(METRIC_PATH).mkdir(parents=True, exist_ok=True)
    save_to = os.path.join(MODEL_PATH, f"{field}.pkl")
    dump(pipe, save_to)

    save_to = os.path.join(METRIC_PATH, f"{field}.meta.json")
    with open(save_to, 'w') as fp:
        json.dump(meta, fp, indent=2)
    print("... done")

    return DescribedComponent(meta, pipe)


def validate(model, field='hate', meta=None):
    import os
    import json

    if meta is None:
        meta = {}

    data = load_data(TEST_SET)
    X = data['text']  # the features we want to analyze
    y = data[field]  # the labels, or answers, we want to test against

    print('... calculate metrics on validation set')
    metrics = calculate_metrics(model, X, y)
    meta['validation'] = metrics

    save_to = os.path.join(METRIC_PATH, f"{field}.meta.json")
    with open(save_to, 'w') as fp:
        json.dump(meta, fp, indent=2)
    print("... done")

    return meta


def train_and_validate(field='hate'):
    import json
    model = train(field)
    meta = validate(model.component, field, model.meta)

    print()
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        field = sys.argv[1]
    else:
        field = 'hate'

    print(f"Running training and validation for `{field}`")

    train_and_validate(field)
