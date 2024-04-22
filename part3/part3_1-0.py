import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    # Convert the 'Date' column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S%z")
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    return df

def limit_data_time_range(df, start_date, end_date):
    # Limit the data to a specific time range
    df = df[start_date:end_date]
    return df

def prepare_data(df):
    # Separate features (X) and target variable (y)
    X = df[['Open', 'High', 'Low', 'Volume']]  
    y = df['Close']
    return X, y

def train_model(X_train, y_train, X_test=None, y_test=None, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    # Create a Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    # Fit the model to the training dataset
    gbr.fit(X_train, y_train)

    # Evaluate on training set
    train_predictions = gbr.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_score = r2_score(y_train, train_predictions)

    # Evaluate on test set if provided
    test_mse, test_score = None, None
    if X_test is not None and y_test is not None:
        test_predictions = gbr.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_score = r2_score(y_test, test_predictions)

    return gbr, train_mse, train_score, test_mse, test_score

def plot_evaluation_metrics(n_estimators_values, train_scores, test_scores, train_mses, test_mses, hyperparameter_name='n_estimators'):
    # Plotting evaluation metrics over hyperparameters
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(n_estimators_values, train_scores, label='Train Score', marker='o')
    plt.plot(n_estimators_values, test_scores, label='Test Score', marker='o')
    plt.title(f'{hyperparameter_name} vs. Model Score')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('R2 Score')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_values, train_mses, label='Train MSE', marker='o')
    plt.plot(n_estimators_values, test_mses, label='Test MSE', marker='o')
    plt.title(f'{hyperparameter_name} vs. Mean Squared Error')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def predict_prices(model, X):
    # Predict future prices on the entire dataset
    predictions = model.predict(X)
    return predictions

def visualize_results(actual_prices, predicted_prices, title='Stock History and Predicted Prices'):
    # Visualize stock history and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices.index, actual_prices.values, label='Actual Prices', color='black')
    plt.plot(actual_prices.index, predicted_prices, label='Predicted Prices', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def visualize_feature_histograms(df):
    # Visualize histograms for each feature
    df.hist(figsize=(12, 8))
    plt.suptitle('Feature Histograms', x=0.5, y=1.02, fontsize=16)
    plt.show()

def visualize_feature_correlation(df):
    # Create a heatmap for feature correlation
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['Open', 'High', 'Low', 'Volume', 'Close']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_feature_importance(importance, names, model_type='Gradient Boosting'):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'{model_type} - Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.show()


# Main program
file_path = 'hist.csv'
start_date = pd.to_datetime('2023-01-01 00:00:00-05:00')
end_date = pd.to_datetime(datetime.now().strftime(f'%Y-%m-%d 00:00:00-05:00'))

# Load data
df = load_data(file_path)

# Limit data to the specified time range
df = limit_data_time_range(df, start_date, end_date)

# Visualize feature histograms
visualize_feature_histograms(df)

# Visualize feature correlation heatmap
visualize_feature_correlation(df)

# Prepare data
X, y = prepare_data(df)

# Split data into training and testing sets
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define hyperparameters to be evaluated
n_estimators_values = [50, 100, 150, 200, 1000]

# Evaluate models over hyperparameters
train_scores, test_scores, train_mses, test_mses = [], [], [], []

for n_estimators_value in n_estimators_values:
    model, train_mse, train_score, test_mse, test_score = train_model(X_train, y_train, X_test, y_test, n_estimators=n_estimators_value)
    train_scores.append(train_score)
    test_scores.append(test_score)
    train_mses.append(train_mse)
    test_mses.append(test_mse)
print("train score"+str(train_scores), "test score"+str(test_scores), "train mse"+str(train_mses), "test mse"+str(test_mses))

# Visualize evaluation metrics over hyperparameters
plot_evaluation_metrics(n_estimators_values, train_scores, test_scores, train_mses, test_mses)

# Train the final model with optimal hyperparameters
optimal_n_estimators = n_estimators_values[test_scores.index(max(test_scores))]
final_model, _, _, _, _ = train_model(X, y, n_estimators=optimal_n_estimators)

# Predict prices
predictions = predict_prices(final_model, X)

# Plot feature importance
plot_feature_importance(final_model.feature_importances_, X.columns)

# Visualize results
visualize_results(y, predictions)