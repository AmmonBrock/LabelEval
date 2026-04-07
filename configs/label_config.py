from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
import os
from pathlib import Path
import yaml

class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    m: float
    n_questions: PositiveInt
    character_1: str
    character_2: str
    judge_model_absolute: str
    feature_labels_absolute: str
    prompt_file: str

    @model_validator(mode="after")
    def __post_init__(self):
        assert self.character_1 != self.character_2, "character_1 and character_2 must be different"
        assert self.m > 0, "m must be a positive float"
        assert self.m <= .5, "m must be less than or equal to 0.5"
        assert os.path.exists(self.judge_model_absolute), f"judge_model_absolute path does not exist: {self.judge_model_absolute}"
        assert os.path.exists(self.feature_labels_absolute), f"feature_labels_absolute path does not exist: {self.feature_labels_absolute}"
        assert self.index_map_path.exists(), f"index_map_path does not exist: {self.index_map_path}"
        assert self.prompt_file_path.exists(), f"prompt_file_path does not exist: {self.prompt_file_path}"

        return self

    @property
    def label_eval_dir(self):
        return Path(__file__).resolve().parent.parent
    
    @property
    def feature_labels_dir(self):
        return Path(self.feature_labels_absolute)
    
    @property
    def index_map_path(self):
        return self.feature_labels_dir / "feature_index_map.json"
    
    @property
    def prompt_file_path(self):
        return self.label_eval_dir / "judge_prompts" / self.prompt_file
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    

