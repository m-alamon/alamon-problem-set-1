'''
PART 1: NETWORK CENTRALITY METRICS

Name: Matthew Alamon
Assignment: INST414 Problem Set 1
'''
import json
import numpy as np
import pandas as pd
import networkx as nx
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

def build_graph(movies):
    """Function to build a graph from a list of movies.
    Args:
        movies (list): List of movies, where each movie is represented as a dictionary.
    Returns:
        networkx.Graph: Graph where nodes are actors and edges represent co-appearances in movies.
    """
    g = nx.Graph()
    for movie in movies:
        actors = movie['actors']

        # Creates a node for every actor.
        for actor_id, actor_name in actors:
            g.add_node(actor_id, name=actor_name)

        # Iterate through the list of actors, generating all pairs.
        for i, (left_actor_id, left_actor_name) in enumerate(actors):
            for right_actor_id, right_actor_name in actors[i+1:]:
                # Get the current weight if exists.
                if g.has_edge(left_actor_id, right_actor_id):
                    g[left_actor_id][right_actor_id]['weight'] += 1
                else:
                    g.add_edge(left_actor_id, right_actor_id, weight=1)
    return g

def calculate_centrality_metrics(g, sample_size=1000):
    """Function to calculate centrality metrics for the graph.
    Args:
        g (networkx.Graph): Graph where nodes are actors and edges represent co-appearances in movies.
        sample_size (int): Number of nodes to sample for approximate calculations.

    Returns:
        pandas.DataFrame: DataFrame containing centrality metrics for each node.
    """
    print("Calculating degree centrality...")
    degree_centrality = nx.degree_centrality(g)
    
    print("Calculating betweenness centrality...")
    betweenness_centrality = nx.betweenness_centrality(g, normalized=True, k=100)  # Approximation with k=100 as the full calculation nearly crashed my computer.
    
    print("Calculating closeness centrality...")
    sample_nodes = np.random.choice(list(g.nodes()), size=min(sample_size, len(g.nodes())), replace=False)
    closeness_centrality = {node: nx.closeness_centrality(g, u=node) for node in sample_nodes} # Approximation with sample size = 1000, as the full calculation nearly crashed my computer.
    
    print("Calculating eigenvector centrality...")
    eigenvector_centrality = nx.eigenvector_centrality(g, max_iter=sample_size, tol=1e-6)
    
    centrality_data = {
        'id': list(range(1, len(g.nodes()) + 1)),
        'actor_id': [],
        'actor_name': [],
        'degree_centrality': [],
        'betweenness_centrality': [],
        'closeness_centrality': [],
        'eigenvector_centrality': []
    }
    
    for node in g.nodes(data=True):
        actor_id = node[0]
        actor_name = node[1]['name']
        centrality_data['actor_id'].append(actor_id)
        centrality_data['actor_name'].append(actor_name)
        centrality_data['degree_centrality'].append(degree_centrality.get(actor_id, 0))
        centrality_data['betweenness_centrality'].append(betweenness_centrality.get(actor_id, 0))
        centrality_data['closeness_centrality'].append(closeness_centrality.get(actor_id, 0))
        centrality_data['eigenvector_centrality'].append(eigenvector_centrality.get(actor_id, 0))
    
    centrality_df = pd.DataFrame(centrality_data)
    return centrality_df

def save_centrality_to_csv(centrality_df):
    """Function to output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`.
    Args:
        centrality_df (pandas.DataFrame): DataFrame containing centrality metrics for each node.
    """
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'network_centrality_{current_datetime}.csv'
    output_path = Path('data') / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centrality_df.to_csv(output_path, index=False)

def main():
    """Main function to load data, build graph, calculate centrality metrics, and save to CSV."""
    filepath = r'C:\Users\Matthew\Documents\INST414\problem-set-1-main\src\imdb_movies_2000to2022.prolific.json'
    movies = load_movies(filepath)
    g = build_graph(movies)
    
    print("Nodes:", len(g.nodes))
    print("Edges:", len(g.edges))
    
    centrality_df = calculate_centrality_metrics(g, sample_size=1000)
    
    save_centrality_to_csv(centrality_df)
    
    # Print the top 10 most central nodes
    top_central_nodes = centrality_df.nlargest(10, 'degree_centrality')
    print("Top 10 most central nodes (by degree centrality):")
    print(top_central_nodes[['actor_name', 'degree_centrality']])

if __name__ == '__main__':
    main()
