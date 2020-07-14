import sys
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    '''
    Fuction loads the data from a sqlite database and creates the X and Y sets.
    
    Input:
        database_filepath - location of the database to use.
        
    Output:
        X - the messages.
        Y - the category data ready for pass through model.
        category_names - column names of rthe Y data.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', 'sqlite:///'  + database_filepath)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    return X, Y


def tokenize(text):
    '''
    This fuction splits the input text into there individual words, removes the stop words, 
    then case normalizes the text to lower case and lemetizes each word in the text, 
    not forgeting the verbs.
    
    Input:
        text - the text you want to subject to the follwoing transformation.
    
    Output:
        clean_tokens - the tokenized and lemmatized text.
    '''
    # Normalize case and remove puctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Initiate lemmatizer, lemmatize and iterate through each token removing leading and 
    # trailing white spaces and remove stop words.
    clean_tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in tokens if w not in stopwords.words('english')]
    
    # Lemmatize verbs by specifying pos. 
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos = 'v').strip() for w in clean_tokens]
    
    
    return clean_tokens


def build_model():
    '''
    Function creates specifies tha model that we will using using the Pipeline function of
    sklearn.
    
    Input:
        None
        
    Output:
        cv - the GridSearchCV of the pipeline for a select parameters.
             
        comment:
            Alternative ouput is pipeline since I don't have enough memory to 
             run cv.
    
    '''
    estimator = [
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ]
    
    pipeline = Pipeline(estimator)

    parameters = {
        #### CountVectorizer ####
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        #'vect__min_df': 1,
        'vect__ngram_range': ((1, 1), (1, 2)),
        
        #### TfidfTransformer ####
        'tfidf__use_idf': (True, False),
        
        #### clf ####
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_estimators': [50, 100, 200]
    }
    
    #cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1)
    
    return pipeline
    
#def evaluate_model(model, X_test, Y_test, category_names):
def evaluate_model(model, X_test, Y_test):
    '''
    This fuction give a report for a multiple column test and preducted data 
    which prints the precision, recall, f1-score and support:
        precision: i.e. what percent of your predictions were correct?
        recall: i.e. what percent of the positve cases did you catch?
        f1-score: i.e. what percent of positive predictions were correct?
    
    Inputs:
       model - This is the model that we are looking to measure.
       Y_test - the test output that we want to compare to predicted.
       Y_pred - the predicted output of X_test.
    
    '''
    Y_pred = model.predict(X_test)
    for i, column in enumerate(Y_test):
        print(column)
        print(classification_report(Y_test[column], Y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        #X, Y, category_names = load_data(database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)
        evaluate_model(model, X_test, Y_test)

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