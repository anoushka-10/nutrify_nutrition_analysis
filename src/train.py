import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your dataset
df = pd.read_csv('data\FOOD-DATA-GROUP4.csv')

# Define feature columns and target
X = df.drop(columns=['Unnamed: 0', 'food', 'Nutrition Density'])
y = df['Nutrition Density']

# Define a threshold to classify 'Nutrition Density'
threshold = 500
y = (y >= threshold).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Save the model and scaler
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
