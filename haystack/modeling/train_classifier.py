import sys
import re
import pandas as pd
import joblib
from sqlalchemy.engine import create_engine
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, coverage_error, f1_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from haystack.modeling.custom_features import QuestionMarkFeature, TextLengthFeature


def load_data(database_filepath : str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('categorized_messages', engine)

    X = df['message'].copy()
    category_names = [col for col in df.columns if col not in ('id', 'message', 'original', 'genre')]
    Y = df[category_names].copy()

    return X, Y, category_names


def tokenize(text : str) -> List[str]:
    """Custom tokenization function. Normalizes, removes urls and stopwords and lemmatizes the input `text`"""
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    # normalize (remove capitalization)
    text = text.lower()
    
    # replace URLs
    text = re.sub('(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'urlplaceholder', text)
    
    # remove punctuation
    text = re.sub('\W', ' ', text)
    
    sentences = sent_tokenize(text)
    
    tokens = []
    for sent in sentences:
        tokens.extend([lemmatizer.lemmatize(word) for word in word_tokenize(sent) if word not in stop_words])
    
    return tokens


def build_model() -> Pipeline:
    """Builds sklearn Pipeline and sets custom parameters found with GridSeachCV on preparation notebook"""

    model = Pipeline([
        ('union', FeatureUnion([
            ('count', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('question', QuestionMarkFeature()),
            ('txtlen', TextLengthFeature())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameters found using GridSearch on preparation notebook
    model_params = {
        'clf__estimator__class_weight': 'balanced',
        'clf__estimator__max_depth': 8,
        'clf__estimator__n_estimators': 300,
        'clf__n_jobs': -1,
        'union__count__tfidf__sublinear_tf': True,
        'union__count__vect__min_df': 1,
        'union__count__vect__ngram_range': (1, 3)
    }

    model.set_params(**model_params)

    return model


def evaluate_model(model, X_test, Y_test, category_names : List[str]) -> None:
    """Performs inference on `X_test` using the provided `model` and prints out precision, recall and f1 metrics for each category"""
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    print('*** Model Evaluation Results ***')
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as out:
        joblib.dump(model, out)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=51)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()