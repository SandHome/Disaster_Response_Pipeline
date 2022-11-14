import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    load data from 02 csv files then merge them to a single pandas dataframe
    
    Input:
    messages_filepath filepath to messages csv file
    categories_filepath filepath to categories csv file
    
    Return:
    output a dataframe merging categories and messages
    '''
    # Get msg from csv file
    msg = pd.read_csv(messages_filepath)
    
    # get categories from csv file
    cg = pd.read_csv(categories_filepath)
    
    # merge categories and msg by id
    output = pd.merge(msg, cg, on="id")
    return output


def clean_data(df):
    '''
    clean_data
    Get dataframe merging categories and messages then clean categories and messages data then return to a dataframe after clean.
    
    Input:
    dataframe merging categories and messages
    
    Return:
    output a dataframe after clean from categories and messages data
    '''
    #  Split categories into separate category columns
    cg = df.categories.str.split(";",expand=True)
    
    # Get columns name after split categories into separate category columns
    r = cg.loc[0,:].apply(lambda x: x[:-2]).values
    
    # Add columns name to dataframe
    cg.columns = r
        
    # Get 0 or 1 value from text value
    for col in cg:
        cg[col] = cg[col].str[-1].astype(int)
    cg["related"] = cg["related"].apply(lambda x: 0 if x ==0 else 1)
    
    # Remove categories columns
    df = df.drop(columns = ["categories"])
    
    # Add data detail of categories to df
    df = pd.concat([df, cg], axis= 1,sort=False)
    
    # Remove duplicates and keep last occurrences of message columns
    df = df.drop_duplicates(subset=['message'], keep='last')
    return df


def save_data(df, database_filename):
    '''
    save_data
    Get dataframe after clean the store to db file
    
    Input:
    df is dataframe after clean
    database_filename is db name
    
    Output
    db file
    '''
    database = sqlalchemy.create_engine('sqlite:///'+ str(database_filename))
    df.to_sql("disastermsg", database, index=False,if_exists='replace')


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