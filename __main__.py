import ecommerce as ec
from utils import wait_for_enter

if __name__ == '__main__':
    df = ec.fetch_data()

    # Print head, info and describe
    print(df.head())
    wait_for_enter()

    print(df.info())
    wait_for_enter()

    print(df.describe())
    wait_for_enter()

    # Create jointplot of 'Time on Website' vs 'Yearly Amount Spent'
    ec.jointplot(
        df,
        x='Time on Website',
        y='Yearly Amount Spent',
        name='time_on_site_vs_yearly'
    )

    # Create jointplot of 'Time on app' vs 'Yearly Amount Spent'
    ec.jointplot(
        df,
        x='Time on App',
        y='Yearly Amount Spent',
        name='time_on_app_vs_yearly'
    )
