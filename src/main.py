'''
Name: Matthew Alamon
Assignment: INST414 Problem Set 1
'''

import json
import os
import requests
import analysis_network_centrality
import analysis_similar_actors_genre

def download_dataset(url, save_path):
    """Download the IMDb movies dataset from the given URL and save it to the specified path.
    Args:
        url (str): URL link from where to download the dataset from.
        save_path (str): Path where the dataset should be saved (should save to \data).
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request failed
    with open(save_path, 'w') as file:
        file.write(response.text)

def main():
    # Defines the paths and URLs.
    dataset_url = 'https://raw.githubusercontent.com/cbuntain/umd.inst414/main/data/imdb_movies_2000to2022.prolific.json'
    dataset_path = os.path.join('data', 'imdb_movies_2000to2022.prolific.json')
    
    # Checks to ensure that the /data directory exists.
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # Downloads and saves the dataset.
    download_dataset(dataset_url, dataset_path)
    
    # Perform network centrality analysis.
    print("Performing network centrality analysis...")
    analysis_network_centrality.main()
    
    # Perform similar actors analysis.
    print("Performing similar actors by genre analysis...")
    query_actor_id = 'nm1165110'  # Example query actor ID (Chris Hemsworth)
    analysis_similar_actors_genre.main(query_actor_id)

if __name__ == "__main__":
    main()
