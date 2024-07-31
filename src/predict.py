import pandas as pd
import joblib
import numpy as np

model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')


df = pd.read_csv('data\FOOD-DATA-GROUP4.csv')

def predict_healthiness(food_name):
    
    food_data = df[df['food'].str.lower() == food_name.lower()]

    if food_data.empty:
        print(f"No data found for '{food_name}'")
        return

    
    features = food_data.drop(columns=['Unnamed: 0', 'food', 'Nutrition Density'])  # Drop non-feature columns
    if features.empty:
        print(f"No nutritional data found for '{food_name}'")
        return
    

    features_standardized = scaler.transform(features)
    
    
    healthiness_pred = model.predict(features_standardized)
    
    
    prediction = "Healthy" if healthiness_pred[0] == 1 else "Unhealthy"
    print(f"Food Item: {food_name}")
    print("Nutritional Values:")
    for column in features.columns:
        print(f"{column}: {features.iloc[0][column]}")
    print(f"Predicted Healthiness: {prediction}")


food_item = input("Enter the food item name: ")
predict_healthiness(food_item)
