import os

import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def save_csv_data(df, path):
    """Save the DataFrame to a .csv file."""
    df.to_csv(path, sep=',', index=False)


def load_data(file_path):    
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path, delimiter=',', decimal='.') 
    return df


def split_data_from_path(data_path, test_size, random_state):
    # Load the data
    df = load_data(data_path)
    return split_data(df=df, test_size=test_size, random_state=random_state)


def split_data(df, test_size, random_state):
    # Perform train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def create_dir_path_results(base_path, dataset, scaling, model, functions, test_size, linear_scaling, pop_size, num_gen, perc_train_records, tournament_size):
    s = base_path.strip()
    test_size = str(test_size).replace('.', 'd')
    linear_scaling = linear_scaling != 0
    perc_train_records = str(perc_train_records).replace('.', 'd')
    functions = '_'.join([ff.strip().lower() for ff in functions])
    s = os.path.join(
        s,
        dataset,
        f'perctrain{perc_train_records}test{test_size}linscale{int(linear_scaling)}',
        f'scaler_{scaling}_model_{model}',
        functions if functions != '' else 'empty',
        f'pop{pop_size}gen{num_gen}toursize{tournament_size}',
        f''
    )

    return s
