
import json
from configs.label_config import LabelConfig
import pandas as pd
import os
from tabulate import tabulate

# Trying to separate labels from the network computations
class LabelRetriever():
    def __init__(self, config: LabelConfig):
        self.config = config
        
    def query_features(self, queries):
        """
        Takes a list of (layer, feature_idx) tuples and returns a Pandas DataFrame 
        containing only the requested features.
        """
        index_map_path = self.config.index_map_path

        # 1. Load the pre-computed index
        with open(index_map_path, 'r') as f:
            # JSON keys are always strings, so we convert them back to ints
            index_map = {int(k): v for k, v in json.load(f).items()}
            
        # 2. Map queries to files to minimize I/O operations
        # Format: { 'path/to/batch-0.jsonl.gz': [feature_idx1, feature_idx2] }
        file_to_queries = {}
        
        for layer, f_idx in queries:
            if layer not in index_map:
                print(f"Warning: Layer {layer} not found in index.")
                continue
                
            found_file = None
            for file_info in index_map[layer]:
                if file_info['min_idx'] <= f_idx <= file_info['max_idx']:
                    found_file = file_info['file_path']
                    break
                    
            if found_file:
                if found_file not in file_to_queries:
                    file_to_queries[found_file] = []
                file_to_queries[found_file].append(f_idx)
            else:
                print(f"Warning: feature {f_idx} not found in layer {layer}")

        # 3. Read the required files and extract the rows
        results = []
        columns_to_keep = ['index', 'layer', 'description', 'typeName', 'explanationModelName']
        
        for file_path, f_indices in file_to_queries.items():
            # Load the specific file
            df = pd.read_json(str(self.config.feature_labels_dir / file_path), lines = True, compression='gzip')
            
            # Filter down to just the requested feature indices
            filtered_df = df[df['index'].isin(f_indices)]
            
            # Filter down columns (handling missing columns gracefully)
            available_cols = [c for c in columns_to_keep if c in filtered_df.columns]
            results.append(filtered_df[available_cols])

        # 4. Combine and return
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
        
    def get_labels_for_neighbors(self, layer, feature_idx, neighbor_results, index_in_sampled = True, additional_label_info = [], show_source_feature = True, show_neighbors = True):
        """Get the feature labels for a list of neighbor results.
        
        Args:
            layer: The layer of the source feature.
            feature_idx: The index of the source feature in the source layer. (This index could be either the original feature index or the index in the sampled features, depending on the value of index_in_sampled.)
            neighbor_results: A list of tuples (layer, feature_idx, weight_value) where feature_idx is either the index in the sampled features or the original features depending on the value of index_in_sampled.
            index_in_sampled: Whether the feature_idx in neighbor_results indicates the sample feature index or not
            additional_label_info: A list of additional label info to include in the returned DataFrame (could be ['typeName', 'explanationModelName'])
            show_source_feature: Whether print the source feature information along with the neighbors
            output: Whether to print the results

        Returns:
            A pandas dataframe containing layer, feature_idx, weight_value, description, and any additional label info specified."""
        
        for c in additional_label_info:
            assert c in ['typeName', 'explanationModelName'], "Additional label info must be in ['typeName', 'explanationModelName']"
        
        index_col_name = "original_feature_idx" if not index_in_sampled else "sampled_feature_idx"
        result = pd.DataFrame(neighbor_results, columns = ["layer", index_col_name, "weight_value"])
        result[["layer", index_col_name]] = result[["layer", index_col_name]].astype(int)
        layers = result['layer']
        feature_indices = result[index_col_name]
        

        # Get the original feature indices
        original_source_feature_idx = self.convert_sample_to_original_index(feature_idx, layer) if index_in_sampled else feature_idx
        original_feature_indices = [self.convert_sample_to_original_index(idx, l) for idx, l in zip(feature_indices, layers)] if index_in_sampled else feature_indices
        if index_in_sampled:
            result['original_feature_idx'] = original_feature_indices

        # Prepare the queries for the feature labels
        queries = list(zip(layers, original_feature_indices))
        queries.append((layer, original_source_feature_idx))

        labels_df = self.query_features(queries)
        labels_df.rename(columns={'index': 'original_feature_idx'}, inplace=True)
        labels_df = labels_df[['original_feature_idx', 'layer', 'description'] + additional_label_info]
        labels_df['layer'] = labels_df['layer'].str.replace("-clt-hp", "").astype(int)


        source_feature_slice = labels_df.loc[labels_df['original_feature_idx'] == original_source_feature_idx]
        labels_df.drop(labels_df[labels_df['original_feature_idx'] == original_source_feature_idx].index, inplace=True)
        
        

        result = result.merge(labels_df, on=['layer','original_feature_idx'], how = 'left')

        # Reorder the columns so that any column that ends in _idx comes first and then the remaining layers
        idx_cols = [col for col in result.columns if col.endswith('_idx')]
        other_cols = [col for col in result.columns if not col.endswith('_idx')]
        result = result[idx_cols + other_cols]
        

        if show_source_feature:
            print(f"Info for feature (Layer {layer}, Feature {feature_idx}):")
            max_col_widths= [None if col != 'description' else 50 for col in source_feature_slice.columns]
            print(tabulate(source_feature_slice, showindex=False, headers="keys", tablefmt="grid",maxcolwidths = max_col_widths ))

        if show_neighbors:
            print(f"Neighbor info:")
            max_col_widths= [None if col != 'description' else 50 for col in result.columns]
            print(tabulate(result, showindex = False, headers="keys", tablefmt="grid",maxcolwidths = max_col_widths))

        return result, source_feature_slice
    
    def examine(self, layer, feature_idx, k=10, index_in_sampled = True, additional_label_info = []):
        if layer < 25:
            print("===== Downstream Neighbors =====")
            downstream_topk = self.get_k_downstream_neighbors(layer, feature_idx, k=k, method = "top", index_in_sampled = index_in_sampled)
            downstream_absbottomk = self.get_k_downstream_neighbors(layer, feature_idx, k=k, method = "abs_bottom", index_in_sampled = index_in_sampled)
            downstream_top_df, _ = self.get_labels_for_neighbors(layer, feature_idx, downstream_topk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = True, show_neighbors = True)
            print("Bottom neighbors")
            downstream_absbottom_df, _ = self.get_labels_for_neighbors(layer, feature_idx, downstream_absbottomk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)

        if layer > 0:
            print("\n===== Upstream Neighbors =====")
            upstream_topk = self.get_k_upstream_neighbors(layer, feature_idx, k=k, method = "top", index_in_sampled = index_in_sampled)
            upstream_absbottomk = self.get_k_upstream_neighbors(layer, feature_idx, k=k, method = "abs_bottom", index_in_sampled = index_in_sampled)
            upstream_top_df, _ = self.get_labels_for_neighbors(layer, feature_idx, upstream_topk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)
            print("Bottom neighbors")
            upstream_absbottom_df, _ = self.get_labels_for_neighbors(layer, feature_idx, upstream_absbottomk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)



        




