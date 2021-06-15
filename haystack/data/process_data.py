import sys
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
import sqlite3
import langdetect
import pycountry as pc
tqdm.pandas()


def load_data(messages_filepath : str, categories_filepath : str):
    """Loads both the messages and categories dfs from .csv files and merges them on the `id` column"""

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id', how='left')

    return df


def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    """
    Description:
        Cleans the merged dataframe `df`. The implemented steps are:
        1. Reformat the `categories` column to represent a numeric encoding of all possible classes
        2. Drop the original `categories` column and replace it with the reformated encoding
        3. Check for duplicated messages and remove them from the dataframe
        4. Drop rows with invalid classes (not in [0, 1])

    Arguments:
        df - The loaded and merged dataframe resulting from "load_data"

    Returns:
        clean_df - The reformatted and clean dataframe
    """

    # Reformat the categories column into separate columns with numerical values
    categories = df['categories'].str.split(';', expand=True)
    
    category_colnames = categories.iloc[0].map(lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.get(-1)
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original 'categories' column and replace with the new multi-column classification target
    clean_df = df.drop('categories', axis=1)
    clean_df = pd.concat([clean_df, categories], axis=1)

    # Check for duplicates and remove them
    n_duplicates = clean_df.duplicated('message').sum()
    if n_duplicates > 0:
        print(f'Dropping {n_duplicates} duplicated rows')
        clean_df = clean_df.drop_duplicates('message', keep='first')
        
        n_duplicates = clean_df.duplicated('message').sum()
        assert n_duplicates == 0, f'Dropping duplicated entries failed. There are still {n_duplicates} duplicates in the data'

    # Drop target cols with invalid values and messages with no content (NaN)
    rows_to_drop = clean_df.message.isna()
    for col in category_colnames:
        rows_to_drop = rows_to_drop | (~clean_df[col].isin([0, 1]))

    print(f'Dropping {rows_to_drop.sum()} rows with invalid (non-binary) category values')
    clean_df = clean_df[~rows_to_drop].copy()
    clean_df

    return clean_df


def save_data(df : pd.DataFrame, database_filename : str, table_name : str ='categorized_messages') -> None:
    """Saves the DataFrame df in the SQLite database located at `database_filename`"""

    # Drop table if it already exists      
    conn = sqlite3.connect(database_filename)
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    conn.close()

    # Save the dataframe in the database
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(table_name, engine, index=False)
    

def detect_language(text : str or None) -> str:
    """
    Detects a given text's language. 
    Returns `None` in case of exceptions or when input is nan, None or an empty string
    """
    
    lang = None
    if str(text).lower() not in ['', 'nan', 'none']:
        try:
            guess = langdetect.detect(text)
            lang = pc.languages.get(alpha_2=guess)
            if lang is not None:
                lang = lang.name

        except Exception as e:
            print(f'TEXT: {text}\n    Raised Exception: {e}')
    
    return lang


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Detecting original languages...')
        df['original_language'] = df.original.progress_map(detect_language)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()