#PASAMOS A PRODUCCION INFROMATION SYSTEM -  RUN COLAB
#!pip install numpy scipy joblib
#!pip install numpy
#! pip install scikit-surprise

import surprise # type: ignore
import numpy
import scipy
import matplotlib
import pandas as pd
import joblib
import os
print("Instalación exitosa.")

# Load the dataset
df = pd.read_csv('C:\Users\perez\Documents\Tesis_USC\ProjectoFinal_IA_USC\src\Project_elections.csv')

# Assuming df is your DataFrame and 'proy_fecha_crea' is the column you want to convert
# Convert the 'proy_fecha_crea' column to datetime
df['proy_fecha_crea'] = pd.to_datetime(df['proy_fecha_crea'], errors='coerce')

from surprise import Dataset, Reader, NormalPredictor, BaselineOnly, KNNWithZScore, SVD
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise.model_selection import train_test_split

# Ensure the data is correctly loaded and prepared into 'df'
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df[['userId', 'proy_pk', 'rating']], reader)
# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Initialize the SVD algorithm
algo = SVD()

# Train the algorithm on the training set
algo.fit(trainset)

#Recommending Projects to a User
# Predict the rating for a specific user and project
user_id = 1  # Replace with the user ID
project_id = 8  # Replace with the project ID
predicted_rating = algo.predict(user_id, project_id)
print(f'Predicted rating for user {user_id} and project {project_id}: {predicted_rating.est}')

#Recommending Top Projects to a User
user_id = 1  # Replace with the user ID
project_ids = df['proy_pk'].unique()  # List of all project IDs

# Predict ratings for all projects the user hasn't rated yet
predictions = []
for project_id in project_ids:
    if df[(df['userId'] == user_id) & (df['proy_pk'] == project_id)].empty:  # If user hasn't rated the project
        predicted_rating = algo.predict(user_id, project_id).est
        predictions.append((project_id, predicted_rating))

# Sort projects by predicted rating in descending order
predictions.sort(key=lambda x: x[1], reverse=True)

# Get the top N project recommendations
top_n = 5  # Number of recommendations
top_recommendations = predictions[:top_n]
print("Top project recommendations for user", user_id)
for project_id, rating in top_recommendations:
    print(f"Project ID: {project_id}, Predicted Rating: {rating}")
# Evaluate the Model 
from surprise import accuracy

# Test the model on the test set
predictions = algo.test(testset)

# Compute and print RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# Save the trained model to a file
model_path = 'svd_recommendation_model.pkl'
joblib.dump(algo, model_path)
print(f"Model saved to {model_path}")

# Load the model from the file
if os.path.exists(model_path):
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully.")
else:
    print("Model file not found.")


    # Treminarl python 7.InformationSystem_appmodelproduccion.py

