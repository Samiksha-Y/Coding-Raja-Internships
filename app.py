import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy
from flask import Flask, request, render_template

# Load the data into a pandas dataframe
ratings = pd.read_csv('Desktop\internship\ratings.csv') 

# Check for missing values
print(ratings.isnull().sum())

# Create a user-item matrix
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

# User-Based Collaborative Filtering
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Evaluate the model
predictions = algo.test(testset)
accuracy.rmse(predictions)

def get_top_n_recommendations(algo, user_id, n=10):
    movie_ids = ratings['movieId'].unique()
    
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    
    predictions = [algo.predict(user_id, movie_id) for movie_id in movie_ids if movie_id not in rated_movies]
    
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n_recommendations = predictions[:n]
    return top_n_recommendations

# Flask web application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = get_top_n_recommendations(algo, user_id, n=10)
    
    return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
