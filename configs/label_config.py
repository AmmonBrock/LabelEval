import json

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
import os
from pathlib import Path
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    eval_name: str
    n_questions: PositiveInt
    character_1: str
    character_2: str
    weight_folder: str #twera, era, virtual_weights, or coact_weights
    use_judge_llm: bool
    validation: bool
    judge_model_absolute: str
    feature_labels_absolute: str
    prompt_file: str
    sample_network_absolute: str
    n_layers: PositiveInt = 26


    @model_validator(mode="after")
    def __post_init__(self):
        assert self.character_1 != self.character_2, "character_1 and character_2 must be different"
        assert os.path.exists(self.judge_model_absolute), f"judge_model_absolute path does not exist: {self.judge_model_absolute}"
        assert os.path.exists(self.feature_labels_absolute), f"feature_labels_absolute path does not exist: {self.feature_labels_absolute}"
        assert self.index_map_path.exists(), f"index_map_path does not exist: {self.index_map_path}"
        assert self.prompt_file_path.exists(), f"prompt_file_path does not exist: {self.prompt_file_path}"

        if not self.weight_folder_path.exists():
            self.prepare_coact_weights()

        assert self.weight_folder_path.exists(), f"weight_folder_path does not exist: {self.weight_folder_path}"


        return self

    def prepare_coact_weights(self):
        if self.weight_folder != "coact_weights":
            return
        if self.weight_folder_path.exists():
            print(f"Weight folder already exists at {self.weight_folder_path}. Skipping preparation.")
            return
        
        coactivations_dir = self.weight_folder_path.parent / "coactivations"
        assert coactivations_dir.exists(), f"coactivations directory does not exist: {coactivations_dir}"

        self.weight_folder_path.mkdir(parents = True, exist_ok = True)
        print("Looping through coactivation stats and preparing coact_weights...")
        for source_layer in range(self.n_layers):
            with safe_open(coactivations_dir / f"coactivation_stats_layer_{source_layer}.safetensors", framework="pt", device = "cpu") as f:
                print(f"Source layer {source_layer}...")
                for target_layer in range(source_layer + 1, self.n_layers):
                    E_ab = f.get_tensor(f"E_ab_{source_layer}_{target_layer}")
                    save_file({f"{source_layer}_{target_layer}": E_ab}, str(self.weight_folder_path / f"{source_layer}_{target_layer}.safetensors"))
                    del E_ab
        return 



    @property
    def label_eval_dir(self):
        return Path(__file__).resolve().parent.parent
    
    @property
    def eval_dir(self):
        return self.label_eval_dir / "results" / self.eval_name
    
    @property
    def feature_labels_dir(self):
        return Path(self.feature_labels_absolute)
    
    @property
    def index_map_path(self):
        return self.feature_labels_dir / "feature_index_map.json"
    
    @property
    def prompt_file_path(self):
        return self.label_eval_dir / "judge_prompts" / self.prompt_file
    
    @property
    def weight_folder_path(self):
        return Path(self.sample_network_absolute) / self.weight_folder
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def lock_parameters(self):
        if (self.eval_dir / "params_lock.json").exists():
            print("Parameters already locked. Checking if they are valid...")
            self.validate_parameters()
            return
        
        params = self.model_dump(exclude = {"eval_name"})
        os.makedirs(str(self.eval_dir), exist_ok = True)
        with open(self.eval_dir / "params_lock.json", "w") as f:
            json.dump(params, f, indent = 4)
        
    
    def validate_parameters(self):
        with open(self.eval_dir / "params_lock.json", "r") as f:
            locked_params = json.load(f)

        config_params = self.model_dump(exclude = {"eval_name"})
        assert locked_params == config_params, f"Current parameters do not match locked parameters. Locked parameters: {locked_params}, current parameters: {config_params}"
        return True
