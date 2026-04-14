
import json
from configs.label_config import LabelConfig
import pandas as pd
import os
from tabulate import tabulate
import numpy as np
from pathlib import Path
from safetensors import safe_open
import torch

# Trying to separate labels from the network computations
class LabelRetriever():
    def __init__(self, config: LabelConfig, use_subnetwork_labels = True):
        self.config = config
        self.root_folder = config.weight_folder_path
        sample_indices_path = Path(self.config.sample_network_absolute) / "sampled_features.npy"
        try:
            self.sample_indices = np.load(str(sample_indices_path))
            assert self.sample_indices is not None, "sample_indices is None. Check if the file exists and is a valid numpy file."
        except Exception as e:
            raise ValueError(f"Error loading sample indices from {sample_indices_path}, {e}")
        
        if use_subnetwork_labels:
            self.prepare_label_loading()
            self.subnetwork_feature_labels = pd.read_csv(config.subnetwork_labels_dir / config.labels_name / "labels.csv", usecols = ["sample_index", "layer", "description"]).set_index(["layer", "sample_index"])['description'].to_dict()
        self.use_subnetwork_labels = use_subnetwork_labels

        
    def prepare_label_loading(self):
        config = self.config
        if (config.subnetwork_labels_dir / config.labels_name / "labels.csv").exists():
            print(f"Subnetwork labels for {config.labels_name} already exist. Skipping generation.")
            if not config.validate_subnetwork_label_params():
                raise ValueError("Subnetwork label parameters do not match locked parameters. Please resolve the mismatch before proceeding.")
            return
        

        sample_indices = self.sample_indices
        layers = np.repeat(np.arange(sample_indices.shape[0]), sample_indices.shape[1])
        original_feature_indices = sample_indices.ravel()
        #Sample feature indices go from 0 to 4999 26 times
        queries = list(zip(layers, original_feature_indices))
        result = self.query_features(queries, iteration = 0)
        result['layer'] = result['layer'].str.replace('-clt-hp', "").astype(int)
        priority_map = {"np_max-act": 1, "np_max-act-logits": 2}
        result['priority'] = result['typeName'].map(priority_map).fillna(3)
        result_sorted = result.sort_values(by=["index", "layer", "priority"])
        result_deduped = result_sorted.drop_duplicates(subset=["index", "layer"], keep = "first")
        layers = result_deduped['layer'].values
        original_feature_indices = result_deduped['index'].values


        # Get the sample_indices in the dataframe as well
        max_val = sample_indices.max()
        reverse_lookup = np.full((sample_indices.shape[0], max_val + 1), -1, dtype=int)
        row_indices = np.arange(sample_indices.shape[0])[:, None]
        col_indices = np.arange(sample_indices.shape[1])
        reverse_lookup[row_indices, sample_indices] = col_indices
        sample_feature_indices = reverse_lookup[layers, original_feature_indices]
        assert -1 not in sample_feature_indices, "Some original feature indices were not found in the sample_indices."
        result_deduped['sample_index'] = sample_feature_indices
        result_deduped.rename(columns={"index": "original_feature_index"}, inplace=True)
        result_deduped = result_deduped[["original_feature_index", "sample_index", "layer", "description", "typeName"]]
        result_deduped['description'] = result_deduped['description'].str[:100] # Truncate descriptions to 100 characters

        os.makedirs(str(config.subnetwork_labels_dir / config.labels_name), exist_ok = True)
        result_deduped.to_csv(config.subnetwork_labels_dir / config.labels_name / "labels.csv", index=False)
        config.lock_subnetwork_label_params()

    def convert_original_to_sample_index(self, original_index, layer):
        sampled_layer_indices = self.sample_indices[layer]
        sample_index = np.where(sampled_layer_indices == original_index)[0]
        if len(sample_index) == 0:
            raise ValueError(f"Original index {original_index} not found in sampled indices for layer {layer}.")
        return int(sample_index[0])  # Return the index in the sampled features
    
    def convert_sample_to_original_index(self, sample_index, layer):
        sampled_layer_indices = self.sample_indices[layer]
        if sample_index < 0 or sample_index >= len(sampled_layer_indices):
            raise ValueError(f"Sample index {sample_index} is out of bounds for layer {layer} with {len(sampled_layer_indices)} sampled features.")
        return int(sampled_layer_indices[sample_index])  # Return the original feature index corresponding to the sample index
    
    def convert_neighbor_results_to_original_indices(self, neighbor_results):
        converted_results = []
        for layer, feature_idx_in_sampled, weight_value in neighbor_results:
            original_feature_idx = self.sample_indices[layer][feature_idx_in_sampled]
            converted_results.append((layer, int(original_feature_idx), weight_value))
        return converted_results
    
    def get_k_downstream_neighbors(self, layer, feature_idx, k=100, method="top", max_layer=None, index_in_sampled = True):
        """Get the k target features most or least influenced by a specific source feature.
        
        Args:
            layer: The source layer of the feature of interest.
            feature_idx: The index of the source feature in the source layer. (This index could be either the original feature index or the index in the sampled features, depending on the value of index_in_sampled.)
            k: The number of top or bottom neighbors to return.
            method: "top" for most positively influenced, "bottom" for most negatively influenced, "abs_bottom" for most influenced regardless of sign.
            max_layer: The maximum layer index to consider as a target (exclusive).
            index_in_sampled: Whether the feature_idx indicates the sample feature index or not
        Returns:
            A list of tuples (target_layer, target_feature_idx, weight_value) for the top k neighbors, where target_feature_idx is the index in either the sampled features or the original features depending on the value of index_in_sampled.
        """
        if max_layer is None:
            max_layer = self.config.n_layers - 1
        assert method in ["top", "bottom", "abs_bottom"], "Method must be 'top', 'bottom', or 'abs_bottom'"

        if not index_in_sampled:
            feature_idx = self.convert_original_to_sample_index(feature_idx, layer) # Convert original feature index to sampled feature index

        slices = []
        target_layers = list(range(layer + 1, max_layer + 1))

        for target_layer in target_layers:
            file_path = str(self.root_folder / f"{layer}_{target_layer}.safetensors")
            tensor_name = f"{layer}_{target_layer}"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing weight file: {file_path}")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                feature_vector = f.get_slice(tensor_name)[feature_idx, :]
                slices.append(feature_vector)

        feature_slice = torch.stack(slices, dim=0)
        flattened_slice = feature_slice.flatten()

        # Create a randomly shuffled version to use for tiebreaking in topk
        rand_idx = torch.randperm(flattened_slice.shape[0])
        shuffled_slice = flattened_slice[rand_idx]
        k = min(k, shuffled_slice.shape[0])  # In case there are fewer than k features total
        if method == "top":
            topk_values, topk_shuffled_indices = torch.topk(shuffled_slice, k=k)
            nonzero_mask = topk_values > 0
            topk_values = topk_values[nonzero_mask]
            topk_shuffled_indices = topk_shuffled_indices[nonzero_mask]
        elif method == "bottom":
            topk_values, topk_shuffled_indices = torch.topk(-shuffled_slice, k=k)
            topk_values = -topk_values
        else:  # abs_bottom
            abs_values = torch.abs(shuffled_slice)
            _, topk_shuffled_indices = torch.topk(-abs_values, k=k)
            topk_values = shuffled_slice[topk_shuffled_indices]

        # Map the shuffled indices back to the original indices
        topk_indices = rand_idx[topk_shuffled_indices]

        layer_offsets = topk_indices // feature_slice.shape[1]
        actual_target_layers = [target_layers[i] for i in layer_offsets.tolist()]
        feature_indices = (topk_indices % feature_slice.shape[1]).tolist()

        result = list(zip(actual_target_layers, feature_indices, topk_values.tolist()))
        if not index_in_sampled:
            result = self.convert_neighbor_results_to_original_indices(result)
        
        return result
        
    def get_k_upstream_neighbors(self, layer, feature_idx, k=100, method="top", min_layer=0, index_in_sampled = True):
        """Get the k source features most or least influencing a specific target feature.
        
        Args:        
            layer: The target layer of the feature of interest.
            feature_idx: The index of the target feature in the target layer. (This index could be either the original feature index or the index in the sampled features, depending on the value of index_in_sampled.)
            k: The number of top or bottom neighbors to return.
            method: "top" for most positively influencing, "bottom" for most negatively influencing, "abs_bottom" for most influencing regardless of sign.
            min_layer: The minimum layer index to consider as a source (inclusive).
            index_in_sampled: Whether the feature_idx indicates the sample feature index or not
        Returns:
            A list of tuples (source_layer, source_feature_idx, weight_value) for the top k neighbors, where source_feature_idx is the index in either the sampled features or the original features depending on the value of index_in_sampled.
        """
        assert method in ["top", "bottom", "abs_bottom"], "Method must be 'top', 'bottom', or 'abs_bottom'"
        if index_in_sampled and self.sample_indices is None:
            raise ValueError("sample_indices must be loaded to use sampled feature indices.")

        if not index_in_sampled:
            feature_idx = self.convert_original_to_sample_index(feature_idx, layer) # Convert original feature index to sampled feature index

        slices = []
        source_layers = list(range(min_layer, layer))

        for source_layer in source_layers:
            file_path = str(self.root_folder / f"{source_layer}_{layer}.safetensors")
            tensor_name = f"{source_layer}_{layer}"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing weight file: {file_path}")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                feature_vector = f.get_slice(tensor_name)[:, feature_idx]
                slices.append(feature_vector)

        feature_slice = torch.stack(slices, dim=0)
        flattened_slice = feature_slice.flatten()

        # Create a randomly shuffled version to use for tiebreaking in topk
        rand_idx = torch.randperm(flattened_slice.shape[0])
        shuffled_slice = flattened_slice[rand_idx]
        k = min(k, shuffled_slice.shape[0])  # In case there are fewer than k features total
        if method == "top":
            topk_values, topk_shuffled_indices = torch.topk(shuffled_slice, k=k)
            nonzero_mask = topk_values > 0
            topk_values = topk_values[nonzero_mask]
            topk_shuffled_indices = topk_shuffled_indices[nonzero_mask]
        elif method == "bottom":
            topk_values, topk_shuffled_indices = torch.topk(-shuffled_slice, k=k)
            topk_values = -topk_values
        else:  # abs_bottom
            abs_values = torch.abs(shuffled_slice)
            _, topk_shuffled_indices = torch.topk(-abs_values, k=k)
            topk_values = shuffled_slice[topk_shuffled_indices]

        # Map the shuffled indices back to the original indices
        topk_indices = rand_idx[topk_shuffled_indices]

        layer_offsets = topk_indices // feature_slice.shape[1]
        actual_source_layers = [source_layers[i] for i in layer_offsets.tolist()]
        feature_indices = (topk_indices % feature_slice.shape[1]).tolist()

        result = list(zip(actual_source_layers, feature_indices, topk_values.tolist()))
        if not index_in_sampled:
            result = self.convert_neighbor_results_to_original_indices(result)
        
        return result
    

    def _compare_to_zero(self, a, method):
        if method == "top":
            return a > 0
        elif method == "bottom":
            return a < 0
        else:  # abs_bottom
            return abs(a) > 0


    def query_features(self, queries, iteration):
        """
        Takes a list of (layer, feature_idx) tuples and returns a Pandas DataFrame 
        containing only the requested features.
        """
        if iteration != 0:
            raise NotImplementedError("Currently only supports querying from the original feature space. Querying from the sampled feature space is not yet implemented.")
        
        index_map_path = self.config.index_map_path

        # 1. Load the pre-computed index
        with open(index_map_path, 'r') as f:
            # JSON keys are always strings, so we convert them back to ints
            index_map = {int(k): v for k, v in json.load(f).items()}
            
        # 2. Map queries to files to minimize I/O operations
        # Format: { 'path/to/batch-0.jsonl.gz': [feature_idx1, feature_idx2] }
        file_to_queries = {}
        
        for idx, (layer, f_idx) in enumerate(queries):
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

        print("Mapped file to queries", flush = True)

        # 3. Read the required files and extract the rows
        results = []
        columns_to_keep = ['index', 'layer', 'description', 'typeName', 'explanationModelName']
        
        print("Num files to read:", len(file_to_queries), flush = True)
        for file_path, f_indices in file_to_queries.items():
            # Load the specific file
            print(file_path, flush = True)

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
        
    def get_labels_for_neighbors(self, layer, feature_idx, neighbor_results, index_in_sampled = True, additional_label_info = [], show_source_feature = True, show_neighbors = True, iteration = 0):
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
        if self.use_subnetwork_labels:
            assert index_in_sampled, "Currently only supports using subnetwork labels when the feature indices are in the sampled feature space. Using original feature indices with subnetwork labels is not yet implemented."
        
        if self.use_subnetwork_labels:
            layers = []
            sample_feature_indices = []
            weight_values = []
            descriptions = []

            for n_layer, f_idx, weight in neighbor_results:
                layers.append(n_layer)
                sample_feature_indices.append(f_idx)
                weight_values.append(weight)
                description = self.subnetwork_feature_labels.get((n_layer, f_idx), None)
                descriptions.append(description)
            return pd.DataFrame({"sampled_feature_idx": sample_feature_indices, "layer": layers, "weight_value": weight_values, "description": descriptions}), pd.DataFrame({"sampled_feature_idx": [feature_idx], "layer": [layer], "description": [self.subnetwork_feature_labels.get((layer, feature_idx), None)]})

        
        print("Deprecation warning: using get_labels_for_neighbors with use_subnetwork_labels = False is deprecated and will be removed in a future version.")
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

        labels_df = self.query_features(queries, iteration = iteration)
        labels_df.rename(columns={'index': 'original_feature_idx'}, inplace=True)
        labels_df = labels_df[['original_feature_idx', 'layer', 'description'] + additional_label_info]
        labels_df['layer'] = labels_df['layer'].str.replace("-clt-hp", "").astype(int)


        source_feature_slice = labels_df.loc[labels_df['original_feature_idx'] == original_source_feature_idx]
        labels_df.drop(labels_df[labels_df['original_feature_idx'] == original_source_feature_idx].index, inplace=True)
        
        

        result = result.merge(labels_df, on=['layer','original_feature_idx'], how = 'left')

        # Reorder the columns so that any column that ends in _idx comes first and then the remaining columns
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
            print("Top neighbors")
            downstream_top_df, _ = self.get_labels_for_neighbors(layer, feature_idx, downstream_topk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = True, show_neighbors = True)
            print("Absolute Bottom neighbors")
            downstream_absbottom_df, _ = self.get_labels_for_neighbors(layer, feature_idx, downstream_absbottomk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)

        if layer > 0:
            print("\n===== Upstream Neighbors =====")
            upstream_topk = self.get_k_upstream_neighbors(layer, feature_idx, k=k, method = "top", index_in_sampled = index_in_sampled)
            upstream_absbottomk = self.get_k_upstream_neighbors(layer, feature_idx, k=k, method = "abs_bottom", index_in_sampled = index_in_sampled)

            print("Top neighbors")
            upstream_top_df, _ = self.get_labels_for_neighbors(layer, feature_idx, upstream_topk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)
            print("Absolute Bottom neighbors")
            upstream_absbottom_df, _ = self.get_labels_for_neighbors(layer, feature_idx, upstream_absbottomk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)