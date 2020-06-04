from ecommerce import fetch_data
from utils import wait_for_enter

if __name__ == '__main__':
    df = fetch_data()

    # Print head, info and describe
    print(df.head())
    wait_for_enter()

    print(df.info())
    wait_for_enter()

    print(df.describe())
    wait_for_enter()
