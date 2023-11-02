import pandas as pd
import numpy as np
import re

from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt


def number_performers(df_concerts, df_concerts1415, df_sub):
    '''
    This function is to calculate the number of perfomers in the next season that has been appeared in the past subscription seasons

    param df_concerts: previous concerts by season
    param df_concerts1415: list of planned concert sets for the 2014-15 season
    param df_sub: previously purchased subscriptions by account
    return: the number of performers in the past 4 seasons for each account.id

    '''

    # Perform data cleaning, replace \r with , in the 'who' column
    df_concerts['who'] = df_concerts['who'].str.replace('\r', ',')

    # Use regular expression to extract the names of the performers
    def extract_names(s):
        # Regular expression pattern to match two consecutive capitalized words
        pattern = r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)+)\b'
        return re.findall(pattern, s)

    df_concerts1415['extracted_names'] = df_concerts1415['who'].apply(extract_names)
    df_concerts['extracted_names'] = df_concerts['who'].apply(extract_names)

    names_list_1415 = np.unique([name for sublist in df_concerts1415['extracted_names'].tolist() for name in sublist]).tolist()

    # get all the season of each unique account.id
    grouped_season_eachID = df_sub.groupby('account.id')['season'].unique().reset_index()

    # Extract all the performers of the past seasons
    df_concerts_extract_names = df_concerts.groupby('season')['extracted_names'].sum().reset_index()
    df_concerts_extract_names['unique_names'] = df_concerts_extract_names['extracted_names'].apply(lambda x: list(set(x)))
    df_concerts_extract_names.drop('extracted_names', axis=1, inplace=True)

    # As we only have the data of the past 4 seasons, we will only consider past 4 seasons here
    df_sub_10_14 = df_sub[df_sub['season'].isin(['2010-2011', '2011-2012', '2012-2013', '2013-2014'])]
    df_sub_10_14 = pd.merge(df_sub_10_14, df_concerts_extract_names, on='season', how='left')

    agg_names = df_sub_10_14.groupby('account.id').apply(
        lambda group: list(set(name for sublist in group['unique_names'] for name in sublist))
    ).reset_index()

    agg_names.columns = ['account.id', 'aggregated_unique_names']

    agg_names['count_in_list'] = agg_names['aggregated_unique_names'].apply(
        lambda names: sum([1 for name in names if name in names_list_1415])
    )

    agg_names_count = agg_names[['account.id', 'count_in_list']]

    return agg_names_count

def tickets_dummy(df_tickets):
    '''
    Perform one-hot encoding to create dummy variables for the top 5 frequent 'price.level','processed_no_seats','multiple.tickets' of each account that has bought tickets in the past

    param df_tickets: previously purchased tickets by account
    return: the number of tickets bought by each account,
            and the dummy variables for the selected features of each account that has bought tickets in the past
    '''

    # Transfer 4.0 to 4 in price.level column
    df_tickets['price.level'] = df_tickets['price.level'].apply(lambda x: str(x).split('.')[0])

    df_tickets_count = df_tickets.groupby(['account.id']).size().reset_index(name='tickets_counts')

    # Sort the dataframe by season and extract unique account.id
    df_tickets['season'] = df_tickets['season'].apply(lambda x: str(x).split('-')[0])
    df_tickets_unique = df_tickets.sort_values('season').drop_duplicates(['account.id'], keep='last')

    # Identify top 5 most frequent values
    N = 5
    no_seats_top_values = df_tickets_unique['no.seats'].value_counts().head(N).index.tolist()
    df_tickets_unique['processed_no_seats'] = df_tickets_unique['no.seats'].apply(lambda x: x if x in no_seats_top_values else 'Other')

    df_tickets_encode = pd.get_dummies(df_tickets_unique, columns=['price.level','processed_no_seats','multiple.tickets'], prefix=['dummy_priceLevel','dummy_seats','dummy_multiple'])
    df_tickets_encode.drop(df_tickets_encode.columns[1:6], inplace=True, axis=1)

    return df_tickets_count, df_tickets_encode

def account_dummy(df_account):
    '''
    Create dummy variables for the top 5 frequent 'shipping.city','relationship' of each account

    param df_account: account information
    return: the dummy variables for the selected features of each account
    '''
    # Identify top N most frequent values
    N = 5
    shipping_cities_top_values = df_account['shipping.city'].value_counts().head(N).index.tolist()
    relationship_top_values = df_account['relationship'].value_counts().head(N).index.tolist()

    # Convert values that are not in the top N to 'Other'
    df_account['processed_shipping_cities'] = df_account['shipping.city'].apply(lambda x: x if x in shipping_cities_top_values else 'Other')
    df_account['processed_relationship'] = df_account['relationship'].apply(lambda x: x if x in relationship_top_values else 'Other')

    # One-hot encode the processed column
    df_encoded = pd.get_dummies(df_account, columns=['processed_shipping_cities', 'processed_relationship'],prefix=['dummy_shipping_city', 'dummy_relationship'])

    return df_encoded

def sub_counting(df_sub):
    '''
    Count the number of subscriptions per account

    param df_sub: previously purchased subscriptions by account
    return: the number of subscriptions per account
    '''
    # Get the number of subscriptions per account
    df_sub_counts = df_sub.groupby('account.id').count()

    return df_sub_counts



def data_preparation(df_train, df_test, df_account, df_tickets, df_sub, df_concerts, df_concerts1415, df_zipcodes):
    '''
    This function is to prepare the data for training and testing
    
    param all the dataframes that from the csv files
    return: the dataframe for training and testing
    '''

    df_encoded = account_dummy(df_account)
    df_tickets_count, df_tickets_encode = tickets_dummy(df_tickets)
    df_sub_counts = sub_counting(df_sub)
    agg_names_count = number_performers(df_concerts, df_concerts1415, df_sub)


    # Target features preparation
    df_all = df_encoded.merge(df_sub_counts, on='account.id', how='left')
    encoded_columns = df_encoded.filter(like='dummy_', axis=1).columns.tolist()
    df_target = df_all[['account.id', 'amount.donated.2013', 'amount.donated.lifetime', 'season']+encoded_columns].fillna(0)
    df_target = df_target.merge(df_tickets_count, on='account.id', how='left').fillna(0)
    df_target = df_target.merge(agg_names_count, on='account.id', how='left').fillna(0)
    # df_target = df_target.merge(df_tickets_encode, on='account.id', how='left').fillna(False)

    # Training data preparation
    df_train_1 = df_train.merge(df_target, on='account.id', how='left')
    df_train_1.drop(['account.id'], inplace=True, axis=1)
    X = df_train_1.loc[:, df_train_1.columns != 'label']
    y = df_train_1['label']

    # split the train data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=18)

    return df_target, X_train, X_val, y_train, y_val

def model_training(X_train, X_val, y_train, y_val):
    '''
    This function is to train the model using XGBoost

    param X_train: the training data
    param X_val: the validation data
    param y_train: the training label
    param y_val: the validation label
    return: the trained model
    '''
    # Convert data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Define the XGBoost parameters
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }

    # Train the XGBoost model
    bst = xgb.train(param, dtrain)

    # Predict on the validation data
    dval = xgb.DMatrix(X_val)
    y_pred = bst.predict(dval)

    # Calculate the AUROC
    auroc = roc_auc_score(y_val, y_pred)
    print(f"AUROC: {auroc:.4f}")

    return bst

def predict(df_test, df_target, bst):
    '''
    This function is to predict the test data using the trained model

    param df_test: the test data
    param df_target: the target features data set
    param bst: the trained model
    '''
    # reset the column 'ID' to 'account.id'
    df_test = df_test.rename(columns={'ID': 'account.id'})
    # predict on the test data

    df_test_1 = df_test.merge(df_target, on='account.id', how='left')
    X_test = df_test_1.loc[:, df_test_1.columns != 'account.id']

    dval_test = xgb.DMatrix(X_test)
    df_test_1['label'] = bst.predict(dval_test)

    # df_test_1['label'] = logreg.predict_proba(df_test_1)[:, 1]
    df_test_1 = df_test_1[['account.id', 'label']]
    df_test_1 = df_test_1.rename(columns={'account.id': 'ID', 'label': 'Predicted'})

    return df_test_1



def main():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    df_account = pd.read_csv('data/account.csv', encoding='latin-1')
    df_tickets = pd.read_csv('data/tickets_all.csv')
    df_sub = pd.read_csv('data/subscriptions.csv')
    df_concerts = pd.read_csv('data/concerts.csv')
    df_concerts1415 = pd.read_csv('data/concerts_2014-15.csv')
    df_zipcodes = pd.read_csv('data/zipcodes.csv')

    df_target, X_train, X_val, y_train, y_val = data_preparation(df_train, df_test, df_account, df_tickets, df_sub, df_concerts, df_concerts1415, df_zipcodes)
    bst = model_training(X_train, X_val, y_train, y_val)
    df_test_1 = predict(df_test, df_target, bst)

    print()
    print("The predicted result is:")
    print(df_test_1.head())
    # df_test_1.to_csv('submission_test.csv', index=False)

if __name__ == "__main__":
    main()

