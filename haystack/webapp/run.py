import json
import plotly
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from haystack.modeling import custom_features

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from haystack.webapp.dataviz import generate_plots

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
db_path = Path(__file__).absolute().parents[1] / 'data/DisasterResponse.db'
model_path = Path(__file__).absolute().parents[1] / 'modeling/classifier.pkl'

engine = create_engine(f'sqlite:///{db_path}')
df = pd.read_sql_table('categorized_messages', engine)

# load model
model = joblib.load(model_path)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    ids, graphJSON = generate_plots(df)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()