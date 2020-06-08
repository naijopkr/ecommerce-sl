import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pandas import DataFrame

ROOT = os.path.abspath(os.curdir)

def fetch_data():
    return pd.read_csv(ROOT + '/data/ecommerce-customers.csv')


def jointplot(
    data: DataFrame,
    x: str,
    y: str,
    name = 'jointplot',
    kind='scatter'
):
    sns.jointplot(x,y,data=data,kind=kind)
    plt.savefig(ROOT + f'/output/{name}.png')


def pairplot(data: DataFrame, name='pairplot'):
    sns.pairplot(data)
    plt.savefig(ROOT + f'/output/{name}.png')


def lmplot(data: DataFrame, x: str, y: str, name='lmplot'):
    sns.lmplot(x,y,data)
    plt.savefig(ROOT + f'/output/{name}.png')
