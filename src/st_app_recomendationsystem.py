import os
import joblib
import pandas as pd
import streamlit as st
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
df = pd.read_csv('C:/Users/perez/Documents/Tesis_USC/ProjectoFinal_IA_USC/src/Project_elections.csv')
df['proy_fecha_crea'] = pd.to_datetime(df['proy_fecha_crea'], errors='coerce')

# Initialize and train the SVD model if not already trained
model_path = 'svd_recommendation_model.pkl'
if os.path.exists(model_path):
    algo = joblib.load(model_path)
    
else:
    # Prepare data for training
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userId', 'proy_pk', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    # Initialize and train the SVD algorithm
    algo = SVD()
    algo.fit(trainset)

    # Evaluate the model
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    st.write(f'RMSE of the model: {rmse:.4f}')

    # Save the model
    joblib.dump(algo, model_path)
    st.write("Model trained and saved successfully.")

# Streamlit UI
st.title('🔮 Project Recommendation System')
st.write("""
This application allows you to:
1. Predict the rating for a specific project for a given user.
2. Recommend top projects for a specific user.
""")

# Sidebar Information
st.sidebar.image('https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/catalogo/iso.png')
st.sidebar.write("""
## Recommendation System
### What is a Recommendation System?
A recommendation system is like a digital assistant that helps you discover things you might like. It looks at your past behavior, preferences, and interactions to suggest new items or content. This could be anything from movies, projects, books, products, music, or even people to connect with.
In this time the recommendation system RMSE: 0.9583.
                 """)

# Recommending Projects to a User
st.header('🔍 Predict Rating for a User and Project')
user_id_input = st.number_input('Enter User ID:', min_value=1, step=1)
project_id_input = st.number_input('Enter Project ID:', min_value=1, step=1)

if st.button('Predict Rating'):
    if user_id_input not in df['userId'].values:
        st.write("User ID not found. Please try again.")
    elif project_id_input not in df['proy_pk'].values:
        st.write("Project ID not found. Please try again.")
    else:
        # Predict the rating for a specific user and project
        predicted_rating = algo.predict(user_id_input, project_id_input)
        st.write(f'Predicted rating for user {user_id_input} and project {project_id_input}: {predicted_rating.est:.2f}')

# Recommending Top Projects to a User
st.header('🌟 Recommend Top Projects to a User')
user_id_input_top = st.number_input('Enter User ID for Recommendations:', min_value=1, step=1)

if st.button('Get Top Project Recommendations'):
    if user_id_input_top not in df['userId'].values:
        st.write("User ID not found. Please try again.")
    else:
        # Predict ratings for all projects the user hasn't rated yet
        project_ids = df['proy_pk'].unique()  # List of all project IDs
        predictions = []
        for project_id in project_ids:
            if df[(df['userId'] == user_id_input_top) & (df['proy_pk'] == project_id)].empty:
                predicted_rating = algo.predict(user_id_input_top, project_id).est
                predictions.append((project_id, predicted_rating))

        # Sort projects by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Get the top N project recommendations
        top_n = 5  # Number of recommendations
        top_recommendations = predictions[:top_n]

        # Display recommendations
        st.write(f"Top {top_n} project recommendations for user {user_id_input_top}:")
        for project_id, rating in top_recommendations:
            st.write(f"Project ID: {project_id}, Predicted Rating: {rating:.2f}")

# Display final note
st.write('---')
st.write('For more information, you can write to us at canarysoftware@gmail.com.')

#For run streamlit desde termianl
# 1.Estar en la carpeta de streamlit cd... por ejemplo en este caso es cd src
# luego de estar en la carpetar ejecutar el comando: streamlit run st_app_recomendationsystem.py