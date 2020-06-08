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

    # Create jointplot with 2D hex bin
    # of 'Time on App' vs 'Length of Membership'
    ec.jointplot(
        df,
        x='Time on App',
        y='Length of Membership',
        name='time_on_app_vs_length_membership',
        kind='hex'
    )


    # Create pairplot of the data set
    ec.pairplot(df)


    # Create a linear model plot (sns.lmplot) of
    # 'Yearly Amount Spent' vs 'Length of Membership'
    # (most correlated features)
    ec.lmplot(
        df,
        'Yearly Amount Spent',
        'Length of Membership',
        'yearly_spent_vs_length_membership'
    )


    # Training and testing data

    # Get training and testing set
    features = [
        'Avg. Session Length',
        'Time on App',
        'Time on Website',
        'Length of Membership'
    ]
    X = df[features]
    y = df['Yearly Amount Spent']
    X_train, X_test, y_train, y_test = ec.train_test_split(X,y,test_size=0.3)

    lm = ec.get_linear_regression(X_train,y_train)

    # Predict off the X_test data set
    predictions = lm.predict(X_test)
    ec.scatterplot(y_test,predictions,name='real_vs_predictions')

    # Evaluate model
    ec.evaluate_model(y_test, predictions)

    # Plot a histogram of the residuals (y_test - predictions)
    ec.distplot(y_test-predictions, name='residuals')

    # Create a dataframe with the coefficients of the linear model
    coefficients = ec.get_coefficients_dataframe(
        lm.coef_,
        features,
        columns=['Coefficients']
    )
    print(coefficients)
    print()
