import json

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
import os
from pathlib import Path
import yaml

class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    eval_name: str
    m: float
    n_questions: PositiveInt
    character_1: str
    character_2: str
    weight_folder: str
    use_judge_llm: bool
    judge_model_absolute: str
    feature_labels_absolute: str
    prompt_file: str
    sample_network_absolute: str
    n_layers: PositiveInt = 26


    @model_validator(mode="after")
    def __post_init__(self):
        assert self.character_1 != self.character_2, "character_1 and character_2 must be different"
        assert self.m > 0, "m must be a positive float"
        assert self.m <= .5, "m must be less than or equal to 0.5"
        assert os.path.exists(self.judge_model_absolute), f"judge_model_absolute path does not exist: {self.judge_model_absolute}"
        assert os.path.exists(self.feature_labels_absolute), f"feature_labels_absolute path does not exist: {self.feature_labels_absolute}"
        assert self.index_map_path.exists(), f"index_map_path does not exist: {self.index_map_path}"
        assert self.prompt_file_path.exists(), f"prompt_file_path does not exist: {self.prompt_file_path}"
        assert self.weight_folder_path.exists(), f"weight_folder_path does not exist: {self.weight_folder_path}"


        return self

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
