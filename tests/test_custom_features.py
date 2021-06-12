from .context import haystack
import numpy as np
from haystack.modeling.custom_features import QuestionMarkFeature, TextLengthFeature
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd

X = pd.Series(
        name='message',
        data=[
            'Lorem ipsum dolor sit amet,', 
            'consectetur adipiscing elit,', 
            'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
            'Ut enim ad minim veniam?', 
            'quis nostrud exercitation ullamco', 
            'laboris nisi ut aliquip ex ea commodo consequat.',
            '???',
            '',
            'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur?',
            'Excepteur sint occaecat cupidatat non proident,', 
            'sunt in culpa qui officia deserunt mollit anim id est laborum.'
        ])

y = np.asarray([
        0, 0, 0, 2, 0, 0, 1, 0, 3, 0, 0
    ]).reshape(-1, 1)

qmf = QuestionMarkFeature()
tlf = TextLengthFeature()

def test_question_mark_fit():
    fit_result = qmf.fit(X, y)

def test_question_mark_fit_transform():
    transform_result = qmf.fit_transform(X, y)
    expected = np.asarray([
        0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0
    ], dtype=np.int32).reshape(-1, 1)

    assert np.equal(transform_result, expected).all()
    assert np.equal(transform_result.shape, expected.shape).all()


def test_question_mark_in_pipeline():
    pipeline = Pipeline(
        [
            ('qmf', QuestionMarkFeature()),
            ('reg', LinearRegression(normalize=True))
        ])
    
    pipeline.fit(X, y)
    pipeline.predict(X)


def test_textlen_fit():
    fit_result = tlf.fit(X, y)


def test_textlen_fit_transform():
    transform_result = tlf.fit_transform(X, y)
    assert np.equal((len(X), 5), transform_result.shape).all()

    