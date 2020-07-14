import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This fuction loads two data sources; one with the messages and the other categories.
    
    Input:
        messages_filepath - file location for the messages data,
        categories_filepath - file location for the categories data.
        
    Output:
        df - merged dataframe of disaster_messages.csv and disaster_categories.csv.
    '''
    # Load the messages and categories csv files.
    messages = pd.read_csv('./' + messages_filepath)
    categories = pd.read_csv('./' + categories_filepath)
    
    # Mearge the two data sources using the id column.
    df = pd.merge(messages, 
                   categories, 
                   on = 'id')
    
    return df


def clean_data(df):
    '''
    Fuction takes in the loaded df and cleans the data.  Spliting the categories into
    columns and converts the category values to 0 or 1, then replaces the old category column
    with the new columns.  Fuction also removes any ducplicates.
    
    Input:
        df - merged dataframe of messageas and categories loaded from csv files.
    
    Output:
        df - the cleaned dataframe.
    '''
    # Select the categories column for cleaning
    categories = df.categories.str.split(';')
    
    # select the first row of the categories dataframe to create headings for the columns
    row = categories[0]
    category_colnames = [i[:-2] for i in row]
    categories = pd.DataFrame(categories.to_list(), 
                              columns = category_colnames)
    
    # Convert the category values to 0 and 1.
    for column in categories:
        categories[column] = categories[column].replace({column: ''}, 
                                                        regex = True)
        categories[column] = categories[column].replace({'-': ''}, 
                                                        regex = True).astype('str')
    
    # join this categories columns to the df dataframe and drop the old categories column.
    df = df.drop(columns = 'categories')
    df = pd.concat([df, categories], 
                   axis = 1)
    
    # Remove any duplicates.
    df.drop_duplicates(keep='first', inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    Fuction saves the cleaned data into sqlite database.
    
    Input:
        df - cleaned dataframe,
        database_filename - selected name for the file.
        
    Output:
        sqlite database
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    

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