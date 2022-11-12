import sys
import pandas as pd
import sqlalchemy

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    # Get data from db file
    database = sqlalchemy.create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql_table('disastermsg', database)
    X = df.message
    Y = df.drop(columns = ["id","message", "genre", "original"])
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    # Write a tokenization function to process your text data
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # Build a machine learning pipeline
    output = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),('clf', MultiOutputClassifier(RandomForestClassifier()))])
    return output


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pre = model.predict(X_test)
    for ind,col in enumerate(category_names):
        print(col, classification_report(Y_test.values[:,ind],Y_pre[:,ind]))


def save_model(model, model_filepath):
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