import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

ROOT = os.path.abspath(os.curdir)

def fetch_data():
    return pd.read_csv(ROOT + '/data/ecommerce-customers.csv')


def scatterplot(x,y,name='scatterplot'):
    plt.scatter(x,y)
    plt.savefig(ROOT + f'/output/{name}.png')
    plt.clf()


def distplot(data: DataFrame, bins=50, name='distplot'):
    sns.distplot(data, bins=bins)
    plt.savefig(ROOT + f'/output/{name}.png')
    plt.clf()


def jointplot(
    data: DataFrame,
    x: str,
    y: str,
    name = 'jointplot',
    kind='scatter'
):
    sns.jointplot(x,y,data=data,kind=kind)
    plt.savefig(ROOT + f'/output/{name}.png')
    plt.clf()


def pairplot(data: DataFrame, name='pairplot'):
    sns.pairplot(data)
    plt.savefig(ROOT + f'/output/{name}.png')
    plt.clf()


def lmplot(data: DataFrame, x: str, y: str, name='lmplot'):
    sns.lmplot(x,y,data)
    plt.savefig(ROOT + f'/output/{name}.png')
    plt.clf()


def get_taining_testing(X,y,test_size=0.4):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=101
    )


def get_linear_regression(X_train,y_train):
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    return lm


def evaluate_model(y_test, predictions):
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print('MAE: ', mae)
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print()


def get_coefficients_dataframe(values, indexes, columns):
    return pd.DataFrame(values, indexes, columns=columns)
