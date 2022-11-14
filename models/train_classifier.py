import sys
import pandas as pd
import numpy as np
import sqlalchemy

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import pickle
import re
from datetime import datetime

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

def load_data(database_filepath):
    '''
    load_data
    get data from db file, split data to X, Y variables then return X, Y and 36 categories name
    
    Input:
    database_filepath filepath to data clean from messages and categories db file
    
    Return:
    X message data
    Y classification results on the other 36 categories
    category_names 36 categories name
    '''
    # Get data from db file
    database = sqlalchemy.create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql_table('disastermsg', database)
    
    X = df.message
    # Get classification results on the other 36 categories
    Y = df.drop(["id","message", "genre", "original"], axis=1)
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    '''
    tokenize
    tokenize the text data
    
    Input:
    text the text data need to tokenize
    
    Return:
    clean_tokens is tokenize results after remove other symbols not is leter and numbers then lemmatize, normalize case, and remove leading/trailing white space
    '''
    # just keep letters and numbers
    tokens = word_tokenize(re.sub("^a-zA-Z0-9"," ", text))
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    Build a machine learning pipeline and use GridSearchCV which is used to find the best parameters for the model
        
    Return:
    machine learning pipeline model 
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])
    # Best Parameters: {'clf__min_samples_split': 2, 'clf__n_estimators': 50, 'features__text_pipeline__vect__ngram_range': (1, 2)}
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }
    # n_jobs=-1 : each core 1 job to speed up ML training
    output = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)

    return output


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    from machine learning pipeline model, train test data then give a Classification Report for each category in category_names
    
    Input
    model machine learning pipeline model
    X_test, Y_test train test data
    category_names is category name list
    
    Return:
    Print out classification Report in sklearn 
     1. Precision: Percentage of correct positive predictions relative to total positive predictions.
     2. Recall: Percentage of correct positive predictions relative to total actual positives.
     3. F1 Score: A weighted harmonic mean of precision and recall. The closer to 1, the better the model.
     F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
     
    and Actual numbers to get best parameters
    '''
    Y_pre = model.predict(X_test)
    
    for ind,col in enumerate(category_names):
        print(col, classification_report(Y_test.values[:,ind],Y_pre[:,ind]))
        
    labels = np.unique(Y_pre)
    accuracy = (Y_pre == Y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
        


def save_model(model, model_filepath):
    '''
    save_model
    store machine learning pipeline model after trained to pickle file
    
    Input
    model machine learning pipeline model after trained
    model_filepath is filepath of pickle file
    
    Return:
    pickle file about trained scikit-learn model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        start = datetime.now()
        print('Start time: ', start.strftime("%m/%d/%Y, %H:%M:%S"))
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        end = datetime.now()
        print('End time: ', end.strftime("%m/%d/%Y, %H:%M:%S"))

        print('Trained model saved after: ', end - start)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()