import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = os.path.abspath(os.curdir)

def fetch_data():
    return pd.read_csv(ROOT + '/data/ecommerce-customers.csv')
