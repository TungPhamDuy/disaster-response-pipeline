import sys
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Load data from SQLite
    Args:
    database_filepath: Path to file sqlite .db
    Return:
    X: Contain message
    Y: Contain the categories
    category_names: List of category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    #df = df[:100]
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    '''
    This function tokenizes a text string into a list of words.
    Args:
    text: A string containing the text to be tokenized.
    Returns:
    A list of tokens (words) extracted from the input text.
    '''
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))),
    ])
    print(pipeline.get_params())
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model
    Args:
    model: Pipeline containing classifier.
    X_test: Series containing messages.
    Y_test: Dataframe containing categories.
    category_names: List of category names.
    '''

    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)

    # Use loop with enumerate for category & metric pairing
    for idx, category in enumerate(category_names):
        print('Category: {}'.format(category))
        print(classification_report(Y_test.iloc[:, idx], Y_pred[category]))
        metrics = {
            'Accuracy': accuracy_score(Y_test.iloc[:, idx], Y_pred[category]),
            'F1 score': f1_score(Y_test.iloc[:, idx], Y_pred[category], average='weighted'),
            'Precision': precision_score(Y_test.iloc[:, idx], Y_pred[category], average='weighted'),
            'Recall': recall_score(Y_test.iloc[:, idx], Y_pred[category], average='weighted')
        }
        for metric_name, metric_value in metrics.items():
            print('{}: {}'.format(metric_name, metric_value))


def save_model(model, model_filepath):
    '''
    Save model to pickle file
    Args:
    model: Pipeline containing classifier.
    model_filepath: Path to pickle file.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        #model.fit(X_train, Y_train)
        parameters = {
            'clf__estimator__n_estimators': [10, 20],
            'clf__estimator__min_samples_split': [2, 4],
        }
        gs = GridSearchCV(model, parameters)
        gs.fit(X_train, Y_train)
        print('Best parameter...')
        print(gs.best_params_)
        
        print('Evaluating model...')
        evaluate_model(gs.best_estimator_, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(gs.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()