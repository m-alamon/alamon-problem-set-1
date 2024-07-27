'''
PART 2: SIMILAR ACTROS BY GENRE
Name: Matthew Alamon
Assignment: INST414 Problem Set 1
'''
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from datetime import datetime
from pathlib import Path

def load_movies(filepath):
        """Function to load movies from a JSON Lines file.
        Args:
        filepath (str): Path to the JSON Lines file containing movie data.
        Returns:
        list: List of movies, where each movie is represented as a dictionary.
        """
        movies = []
        with open(filepath, 'r') as in_file:
                for line in in_file:
                        try:
                                movie = json.loads(line.strip())
                                movies.append(movie)
                        except json.JSONDecodeError as e:
                                print(f"Error decoding JSON: {e}")
        return movies

def build_genre_matrix(movies):
        """Function to build a dataframe matrix with a list of genres based off of movies.
        Args:
        movies (list): List of movies, where each movie is represented as a dictionary.
        Returns:
        pandas.DataFrame: DataFrame where each row corresponds to an actor,
        each column represents a genre, and each cell captures
        how many times that actor has appeared in that genre.
        """
        genre_dict = {}
        all_genres = set()

        # Update to collect all unique genres.
        for movie in movies:
                all_genres.update(movie['genres'])

        for movie in movies:
                genres = movie['genres']
                for actor_id, actor_name in movie['actors']:
                        if actor_id not in genre_dict:
                                genre_dict[actor_id] = {'actor_name': actor_name}
                                for genre in all_genres:
                                        genre_dict[actor_id][genre] = 0
                        for genre in genres:
                                genre_dict[actor_id][genre] += 1

        genre_df = pd.DataFrame.from_dict(genre_dict, orient='index').fillna(0)
        genre_df.reset_index(inplace=True)
        genre_df.rename(columns={'index': 'actor_id'}, inplace=True)
        return genre_df

def find_similar_actors(genre_df, query_actor_id, metric='cosine'):
        """Function to find the top 10 most similar actors to the query actor based on genre appearances.
        Args:
        genre_df (pandas.DataFrame): DataFrame where each row corresponds to an actor and each column represents a genre.
        query_actor_id (str): Actor ID of the query actor.
        metric (str): Distance metric to use ('cosine' or 'euclidean').
        Returns:
        pandas.DataFrame: DataFrame containing the top 10 most similar actors to the query actor.
        """
        actor_ids = genre_df['actor_id']
        genre_features = genre_df.drop(columns=['actor_id', 'actor_name'])

        query_index = genre_df.index[genre_df['actor_id'] == query_actor_id].tolist()[0]
        query_vector = genre_features.iloc[query_index].values.reshape(1, -1)
    
        if metric == 'cosine':
                distances = cosine_similarity(genre_features, query_vector).flatten()
        elif metric == 'euclidean':
                distances = euclidean_distances(genre_features, query_vector).flatten()
    
        genre_df['distance'] = distances
        top_similar_actors = genre_df.nlargest(10, 'distance' if metric == 'cosine' else 'distance', keep='all')

        return top_similar_actors[['actor_id', 'actor_name', 'distance']]

def save_similar_actors_to_csv(similar_actors_df, query_actor_id):
        """Function to save similar actors to a CSV file.
        Args:
        similar_actors_df (pandas.DataFrame): DataFrame containing the top 10 most similar actors.
        query_actor_id (str): Actor ID of the query actor.
        """
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'similar_actors_genre_{query_actor_id}_{current_datetime}.csv'
        output_path = Path('data') / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        similar_actors_df.to_csv(output_path, index=False)

def main(query_actor_id):
        """Main function to load data, build genre matrix, find similar actors, and save to CSV."""
        filepath = r'C:\Users\Matthew\Documents\INST414\problem-set-1-main\src\imdb_movies_2000to2022.prolific.json'
        movies = load_movies(filepath)
    
        genre_df = build_genre_matrix(movies)

        similar_actors_cosine_df = find_similar_actors(genre_df, query_actor_id, metric='cosine')
        similar_actors_cosine_df['query_actor_id'] = query_actor_id
        similar_actors_euclidean_df = find_similar_actors(genre_df, query_actor_id, metric='euclidean')
        similar_actors_euclidean_df['query_actor_id'] = query_actor_id

        save_similar_actors_to_csv(similar_actors_cosine_df, query_actor_id) #Saves cosine similarity results to a CSV.
    
        print("Top 10 most similar actors (by genre appearances, cosine distance):")
        print(similar_actors_cosine_df[['actor_id', 'actor_name', 'distance']])
    
        cosine_top_10 = set(similar_actors_cosine_df['actor_name'])
        euclidean_top_10 = set(similar_actors_euclidean_df['actor_name'])
    
        print("\nDescription of changes based on Euclidean distance:")
        print("Using Euclidean distance instead of cosine similarity would alter the results because Euclidean distance weighs the magnitude of genre appearances.")
        print("This means actors who have starred in a large number of movies in the same gernres as the requested actor will have a smaller Euclidean distance. Cosine similarity, however, focusing on the direction of the genre appearances,") 
        print("givies more importance to the relative distribution across genres rather than the absolute counts like Euclidean. Euclidean distance might favor actors with a wider range of genres, making it potentially inaccurate based on what you are looking for.")
        print(" ")
        only_cosine = cosine_top_10 - euclidean_top_10
        only_euclidean = euclidean_top_10 - cosine_top_10
        common = cosine_top_10 & euclidean_top_10
        print(f"For instance, these are actors that uniquely appear in the cosine top 10: {only_cosine}")
        print(f"While these are the actors that uniquely appear in the Euclidean top 10: {only_euclidean}")
        print(f"Actors present in both top 10 lists: {common}")

if __name__ == '__main__':
        main()
