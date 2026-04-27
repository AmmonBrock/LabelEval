import shelve
from configs.label_config import LabelConfig
import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd




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