import sys
import pandas as pd
import numpy as np    
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from csv files and merge them into a single dataframe 
    Args:
    messages_filepath: path to messages csv file
    categories_filepath: path to categories csv file
    Return:
    df: the loaded df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left = messages, right = categories, on = 'id', how = 'inner')
    return df

def clean_data(df):
    '''
    Clean the loaded df
    Args:
    df: the loaded df
    Return:
    df: the clean df
    '''
    # Split categories into seperate column
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]
    categories_column = [category.split('-')[0] for category in row]
    categories.columns = categories_column
    # Convert categories into binary number
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    # Replace categories column in df
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    Save df to sqlite db as .db
    Args: 
    df: the cleaned df
    database_filename: the assigned namefile of database 
    Return: .db file save in sqlite
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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