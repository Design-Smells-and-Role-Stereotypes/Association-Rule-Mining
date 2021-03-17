import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np


def data_prep(data):
    # read data
    df = pd.read_csv(data)
    df = df.iloc[:, 3:21]

    # perform one-hot encoding on "label"
    df_ = pd.get_dummies(df, prefix='_')
    return df_


def rules_mining(df):

    # looking at the association of role-stereotypes with design smells
    # build the model
    # you can adjust the parameters accordingly
    assoc = apriori(df, min_support=0.05, use_colnames=True)
    # collecting the inferred rules in a dataframe
    rules = association_rules(assoc, metric="lift")
    rules = rules.sort_values(
        ['confidence', 'conviction'], ascending=[False, False])
    rules.to_csv(r'result/assoc_rules.csv')
    print(rules)


if __name__ == "__main__":
    data = 'data/all_labeled_data_clustered.csv'
    dp = data_prep(data)
    rules_mining(dp)
