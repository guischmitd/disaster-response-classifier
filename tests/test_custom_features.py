from .context import haystack
import numpy as np
from haystack.modeling.custom_features import QuestionMarkFeature, TextLengthFeature
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample data for testing
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

# initializing custom transformers
qmf = QuestionMarkFeature()
tlf = TextLengthFeature()

def test_question_mark_fit():
    """Tests call to transformer fit method"""
    fit_result = qmf.fit(X, y)

def test_question_mark_fit_transform():
    """Tests QuestionMark features on sample data and compares results to expected"""

    transform_result = qmf.fit_transform(X, y)
    expected = np.asarray([
        0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0
    ], dtype=np.int32).reshape(-1, 1)

    assert np.equal(transform_result, expected).all()
    assert np.equal(transform_result.shape, expected.shape).all()


def test_question_mark_in_pipeline():
    """Tests calls to qmf fit and predict (fit_transform and transform) in a Pipeline"""
    pipeline = Pipeline(
        [
            ('qmf', QuestionMarkFeature()),
            ('reg', LinearRegression(normalize=True))
        ])
    
    pipeline.fit(X, y)
    pipeline.predict(X)


def test_textlen_fit():
    """Tests call to transformer fit method"""

    fit_result = tlf.fit(X, y)


def test_textlen_fit_transform():
    """Tests TextLength features on sample data and compares results to expected"""
    transform_result = tlf.fit_transform(X, y)
    
    expected = np.asarray([
        [23, 6, 1, 0, 3.83333333],
        [26, 4, 1, 0, 6.5],
        [56, 12, 1, 1, 4.66666667],
        [20, 6, 1, 0, 3.33333333],
        [30, 4, 1, 0, 7.5],
        [41, 9, 1, 0, 4.55555556],
        [3, 3, 2, 0, 1],
        [0, 0, 0, 0, 0],
        [87, 17, 1, 2, 5.11764706],
        [42, 7, 1, 0, 6],
        [52, 12, 1, 1, 4.33333333]
        ])

    assert np.equal((len(X), 5), transform_result.shape).all(), "Transformed and expected shapes do not match"
    assert np.isclose(expected, transform_result).all(), "Transformed values are not within tolerance of expected"


def test_textlen_in_pipeline():
    """Tests calls to tlf fit and predict (fit_transform and transform) in a Pipeline"""
    pipeline = Pipeline(
        [
            ('tlf', TextLengthFeature()),
            ('reg', LinearRegression(normalize=True))
        ])
    
    pipeline.fit(X, y)
    pipeline.predict(X)