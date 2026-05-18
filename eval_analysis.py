import shelve
from configs.label_config import LabelConfig
import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import re




def scores_per_layer(config: LabelConfig):
    stream = "downstream"
    with shelve.open(str(config.eval_dir / "scores"), "r") as db:
        all_layer_scores = {}
        for layer in range(config.n_layers):
            layer_scores = []
            for f_idx in range(5000):
                key = f"{layer}_{f_idx}_{stream}"
                if key in db:
                    layer_scores.append(db[key]["score"])
            all_layer_scores[layer] = layer_scores

    return all_layer_scores

def plot_average_score_per_layer(config, all_layer_scores):
    average_scores = {layer: sum(scores) / len(scores) if scores else 0 for layer, scores in all_layer_scores.items()}
    layers = list(average_scores.keys())
    avg_scores = list(average_scores.values())

    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_scores, marker='o')
    plt.title('Average Score per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Average Score')
    plt.xticks(layers)
    plt.grid()
    os.makedirs(str(config.eval_dir / "images"), exist_ok = True)
    plt.savefig(str(config.eval_dir / "images" / "average_score_per_layer.png"))
    plt.close()

def plot_score_distribution_per_layer(config, all_layer_scores):
    # Make a histogram for each layer's scores across many subplots. Use 5 rows and as many columns as needed.
    n_layers = len(all_layer_scores)
    n_cols = (n_layers + 4) // 5
    plt.figure(figsize=(5 * n_cols, 25))
    for i, (layer, scores) in enumerate(all_layer_scores.items()):
        plt.subplot(5, n_cols, i + 1)
        plt.hist(scores, bins=20, alpha=0.7)
        plt.title(f'Layer {layer} Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid()
    os.makedirs(str(config.eval_dir / "images"), exist_ok = True)
    plt.tight_layout()
    plt.savefig(str(config.eval_dir / "images" / "score_distribution_per_layer.png"))
    plt.close()

def view_extreme_scorers(config, downstream = True, high = True, threshold = .1):
    stream = "downstream" if downstream else "upstream"
    scores = []
    layers = []
    indices = []
    with shelve.open(str(config.eval_dir / "scores"), "r") as db:
        # Implementation for viewing extreme scorers
        for key in db.keys():
            if stream in key:
                layer, f_idx, _ = key.split("_")
                score = db[key]["score"]
                scores.append(score)
                layers.append(int(layer))
                indices.append(int(f_idx))
    
    scores = np.array(scores)
    layers = np.array(layers)
    indices = np.array(indices)

    insane_mask = scores > (1 - threshold) if high else scores < threshold
    insane_scores = scores[insane_mask]
    insane_layers = layers[insane_mask]
    insane_indices = indices[insane_mask]

    labels_df = pd.read_csv(config.subnetwork_labels_dir / config.labels_name / "labels.csv")
    filter_df = pd.DataFrame({'sample_index': insane_indices, 'layer': insane_layers, 'score': insane_scores})
    merged_df = pd.merge(labels_df, filter_df, on = ['sample_index', 'layer'], how = 'inner')

    return merged_df.sort_values(by = "score", ascending = not high)

def plot_layer_score_scatter(config, all_layer_scores):
    # Create a scatter plot of scores for each layer
    plt.figure(figsize=(10, 6))
    for layer, scores in all_layer_scores.items():
        plt.scatter([layer] * len(scores), scores, alpha=0.5)
    plt.title('Score Scatter Plot per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.xticks(list(all_layer_scores.keys()))
    plt.grid()
    os.makedirs(str(config.eval_dir / "images"), exist_ok = True)
    plt.savefig(str(config.eval_dir / "images" / "score_scatter_per_layer.png"))
    plt.close()

def plot_layer_score_boxplot(config, all_layer_scores):
    # Create a boxplot of scores for each layer
    plt.figure(figsize=(10, 6))
    layers = list(all_layer_scores.keys())
    scores = [all_layer_scores[layer] for layer in layers]
    plt.boxplot(scores, tick_labels=layers)
    plt.title('Score Boxplot per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.xticks(layers)
    plt.grid()
    os.makedirs(str(config.eval_dir / "images"), exist_ok = True)
    plt.savefig(str(config.eval_dir / "images" / "score_boxplot_per_layer.png"))
    plt.close()


def computational_intermediate_analysis(config):
    all_scores = view_extreme_scorers(config, threshold = 1.) # Gets the scores for all features
    # We remove Oklahoma from the list of states because its capital contains the name of the state which defeats the purpose of using a computational intermediate
    states = [
        "Alabama", "Alaska", "Arizona", "California", "Colorado", 
        "Connecticut", "Florida", "Georgia", "Illinois", "Iowa", 
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
        "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", 
        "Nebraska", "Nevada", "New Jersey", "New Mexico", "New York", 
        "North Carolina", "North Dakota", "Ohio", "Oregon", "Pennsylvania", 
        "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
        "Washington", "Wisconsin", "Wyoming"
    ]

    ### For later
    # state_capitals = [
    #     "Montgomery", "Juneau", "Phoenix", "Sacramento", "Denver", 
    #     "Hartford", "Tallahassee", "Atlanta", "Springfield", "Des Moines", 
    #     "Topeka", "Frankfort", "Baton Rouge", "Augusta", "Annapolis", 
    #     "Lansing", "St. Paul", "Jackson", "Jefferson City", "Helena", 
    #     "Lincoln", "Carson City", "Trenton", "Santa Fe", "Albany", 
    #     "Raleigh", "Bismarck", "Columbus", "Salem", "Harrisburg", 
    #     "Pierre", "Nashville", "Austin", "Salt Lake City", "Montpelier", 
    #     "Olympia", "Madison", "Cheyenne"
    # ]
    # state_cities = [
    #     "Birmingham", "Anchorage", "Tucson", "Los Angeles", "Boulder", 
    #     "New Haven", "Miami", "Savannah", "Chicago", "Cedar Rapids", 
    #     "Wichita", "Louisville", "New Orleans", "Bangor", "Baltimore", 
    #     "Detroit", "Minneapolis", "Biloxi", "St. Louis", "Bozeman", 
    #     "Omaha", "Las Vegas", "Princeton", "Albuquerque", "Buffalo", 
    #     "Charlotte", "Fargo", "Cleveland", "Eugene", "Philadelphia", 
    #     "Sioux Falls", "Memphis", "Dallas", "Provo", "Burlington", 
    #     "Seattle", "Milwaukee", "Laramie"
    # ]

    countries = [
        "Australia", "Belgium", "Brazil", "Canada", "China", 
        "Colombia", "Croatia", "Ecuador", "France", "Germany", 
        "Greece", "India", "Iran", "Iraq", "Israel", 
        "Italy", "Japan", "Kazakhstan", "Kenya", "Morocco", 
        "Myanmar", "Netherlands", "New Zealand", "Nigeria", "Norway", 
        "Pakistan", "Peru", "Philippines", "Poland", "Portugal", 
        "Saudi Arabia", "South Africa", "South Korea", "Spain", "Sweden", 
        "Switzerland", "Tanzania", "Turkey", "United Arab Emirates", "United Kingdom", 
        "United States", "Vietnam"
    ]
    # country_capitals = [
    #     "Canberra", "Brussels", "Brasilia", "Ottawa", "Beijing", 
    #     "Bogota", "Zagreb", "Quito", "Paris", "Berlin", 
    #     "Athens", "New Delhi", "Tehran", "Baghdad", "Jerusalem", 
    #     "Rome", "Tokyo", "Astana", "Nairobi", "Rabat", 
    #     "Naypyidaw", "Amsterdam", "Wellington", "Abuja", "Oslo", 
    #     "Islamabad", "Lima", "Manila", "Warsaw", "Lisbon", 
    #     "Riyadh", "Pretoria", "Seoul", "Madrid", "Stockholm", 
    #     "Bern", "Dodoma", "Ankara", "Abu Dhabi", "London", 
    #     "Washington, D.C.", "Hanoi"
    # ]

    # country_cities = [
    #     "Sydney", "Antwerp", "Rio de Janeiro", "Toronto", "Shanghai", 
    #     "Medellin", "Dubrovnik", "Guayaquil", "Marseille", "Munich", 
    #     "Thessaloniki", "Mumbai", "Isfahan", "Basra", "Tel Aviv", 
    #     "Milan", "Osaka", "Almaty", "Mombasa", "Casablanca", 
    #     "Yangon", "Rotterdam", "Auckland", "Lagos", "Bergen", 
    #     "Karachi", "Cusco", "Cebu", "Krakow", "Porto", 
    #     "Jeddah", "Johannesburg", "Busan", "Barcelona", "Gothenburg", 
    #     "Geneva", "Dar es Salaam", "Istanbul", "Dubai", "Liverpool", 
    #     "Chicago", "Ho Chi Minh City"
    # ]

    countries_and_states = states + countries

    pattern = r'\b(?:' + '|'.join(countries_and_states) + r')\b'
    replacement_candidates = all_scores.loc[all_scores.description.str.contains(pattern, case = False, na = False, regex = True)].copy()


    # Add entity_name and country columns
    standard_name_map = {entity.lower(): entity for entity in countries_and_states}
    is_country_map = {entity.lower(): (entity in countries) for entity in countries_and_states}
    sorted_entities = sorted(countries_and_states, key=len, reverse=True)
    extract_pattern = r'\b(' + '|'.join(sorted_entities) + r')\b'
    extracted_matches = replacement_candidates['description'].str.extract(extract_pattern, flags=re.IGNORECASE, expand=False)
    replacement_candidates['entity_name'] = extracted_matches.str.lower().map(standard_name_map)
    replacement_candidates['country'] = extracted_matches.str.lower().map(is_country_map)

    #After reviewing the descriptions manually, I decided to remove features that contained multiple unrelated concepts, multiple states, or only referred to a city
    to_remove = ["India and China", 'Mentions "means" or "New York"', "Hungary and Belgium", "New York City", "COVID-19 and Australia", "Alaska or Finland", "Washington D.C.", "Canada and Licenses", "United States and charts/numbers", "New York City, cities", "North Carolina, New South Wales", "Denmark and Germany names/places", "Wisconsin, universities, geography", "Alaska Native/Canadian", "Pennsylvania, education, therapy", "Louisiana and PHP code", 'Mathematica, Utah, "ica"', "New Zealand, code", "Locations/Maryland/Virginia", "Bond/Ecuador/bonds", "United Arab Emirates and NYC", "Locations in Washington/Oregon", "HTML and California", "code, definitions, Nigeria"]
    replacement_candidates = replacement_candidates.loc[~replacement_candidates.description.isin(to_remove)]





def get_scores_for_features(config, layers, original_indices, downstream = True):
    from label_evaluation import LabelEvaluation
    le = LabelEvaluation(config)
    for layer, orig_idx in zip(layers, original_indices):
        try:
            sample_index = le.weight_network.convert_original_to_sample_index(orig_idx, layer)
            print(f"Feature on layer {layer} at original index {orig_idx}: {le.get_evaluation_result(layer, sample_index, downstream)}")
        except:
            print(f"Skipping layer {layer} index {orig_idx}")
            continue

        




def main(config: LabelConfig):
    all_layer_scores = scores_per_layer(config)
    plot_average_score_per_layer(config, all_layer_scores)
    plot_score_distribution_per_layer(config, all_layer_scores)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyze feature evaluations")
    parser.add_argument("--config", help="Path to the label config file")
    args = parser.parse_args()
    config = LabelConfig.from_yaml(os.path.join("configs", args.config))
    main(config)