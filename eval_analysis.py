import shelve
from configs.label_config import LabelConfig
import argparse
from matplotlib import pyplot as plt
import os




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