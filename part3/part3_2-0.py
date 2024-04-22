import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data from the CSV file
data = pd.read_csv('bank.csv', delimiter=';')

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Split the data into features (X) and target variable (y)
X = data.drop('y', axis=1)
y = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vary the number of estimators
n_estimators_list = [50, 100, 150, 200, 1000]
accuracy_scores = []

for n_estimators in n_estimators_list:
    # Create and train the Gradient Boosting classifier
    gb_classifier = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1, max_depth=3, random_state=42)
    gb_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = gb_classifier.predict(X_test)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # Print classification report
    report = classification_report(y_test, y_pred)
    print(f"Classification Report (n_estimators={n_estimators}):\n{report}")

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (n_estimators={n_estimators})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plotting accuracy as a line chart
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, accuracy_scores, marker='o')
plt.title('Accuracy vs. Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
