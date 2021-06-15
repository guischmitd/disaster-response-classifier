from typing import Any, List, Tuple
import plotly
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy.engine import base
import json

def generate_plots(df : pd.DataFrame) -> Tuple[List[int], List[dict]]:
    """Generates descriptive plots of the training data for frontend"""

    base_layout = {
        'plot_bgcolor': '#212529',
        'paper_bgcolor':"#212529",
        'font': {
            # 'family': 'Helvetica',
            # 'size': 16,
            'color': '#f8f9fa',
        }
    }

    y = df[[c for c in df.columns if c not in ['id', 'original', 'original_language', 'message', 'genre']]]
    graphs = []

    # 1. Messages per category
    values = y.sum().sort_values(ascending=False)
    labels = values.index.str.replace('_', ' ').str.capitalize()
    
    graphs.append({
        'data': [go.Bar(x=labels, y=values, marker_color='#198754')],
        'layout': generate_layout_from_template(
            base_layout, 
            title='Messages per category', 
            xaxis={'title': 'Category'},
            yaxis={'title': 'Message count'})
        })

    # 2. Multilabel messages
    cats_per_sample = y.sum(axis=1)
    cats_per_sample[cats_per_sample <= 10]

    graphs.append({
        'data': [go.Histogram(x=cats_per_sample[cats_per_sample <= 10], marker_color='#198754')],
        'layout': generate_layout_from_template(
            base_layout, 
            title='Multilabeled messages', 
            xaxis={'title': 'Number of categories'},
            yaxis={'title': 'Message count'})
        })

    # 3. Non-english language distribution
    lang_counts = df.original_language.value_counts().sort_values(ascending=True).drop('English')

    graphs.append({
        'data': [go.Bar(x=lang_counts, y=lang_counts.index, orientation='h', marker_color='#198754')],
        'layout': generate_layout_from_template(
            base_layout,
            height=800,
            title='Non-english original language distribution',
            xaxis={'title': 'Message count'},
            yaxis={'title': 'Language', 'dtick': 1})
        })
        
    # 4. Message length distribution (related vs unrelated)
    message_lens = df.groupby('related').apply(lambda x: x['message'].str.len())


    graphs.append({
        'data': [
            go.Histogram(x=message_lens[1], name='Related', marker_color='#198754'),
            go.Histogram(x=message_lens[0], name='Unrelated', marker_color='#dc3444')
            ],
        'layout': generate_layout_from_template(
            base_layout, 
            title='Related/Unrelated message character count distribution', 
            xaxis={
                'title': 'Character count',
                'range': [30, 800]
                },
            yaxis={'title': 'Message count'},
            barmode='overlay')
        })
        

    # 5. Message counts by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    graphs.append({
        'data': [go.Bar(x=genre_names, y=genre_counts, marker_color='#198754')],
        'layout': generate_layout_from_template(
            base_layout, 
            title='Message genre distribution', 
            xaxis={'title': 'Genre'},
            yaxis={'title': 'Message count'},
            )
        })
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON


def generate_layout_from_template(template : dict, **kwargs) -> dict:
    """Helper function that returns updated dicts from a base template and kwargs. Used mainly to avoid code repetition"""
    
    new_layout = template.copy()
    new_layout.update(**kwargs)

    return new_layout