"""
配置管理模块
支持训练、模型、数据等各种配置
"""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: Optional[str] = None
    cache_dir: Optional[str] = "./models"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_sequence_length: int = 2048


@dataclass
class DataConfig:
    """数据配置"""
    train_file: str = "data/train.jsonl"
    val_file: Optional[str] = "data/val.jsonl"
    test_file: Optional[str] = "data/test.jsonl"
    input_column: str = "input"
    output_column: str = "output"
    max_input_length: int = 1024
    max_output_length: int = 1024
    data_format: str = "jsonl"  # jsonl, json, csv
    preprocessing_num_workers: int = 4


@dataclass  
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: List[str] = None
    run_name: Optional[str] = None
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True
    
    def __post_init__(self):
        if self.report_to is None:
            self.report_to = []


@dataclass
class InferenceConfig:
    """推理配置"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = None
    batch_size: int = 8
    max_eval_samples: Optional[int] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["bleu", "rouge", "perplexity"]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.training_config = TrainingConfig()
        self.inference_config = InferenceConfig()
        self.evaluation_config = EvaluationConfig()
        
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """从文件加载配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 更新配置
        if 'model' in config_dict:
            self._update_config(self.model_config, config_dict['model'])
        if 'data' in config_dict:
            self._update_config(self.data_config, config_dict['data'])
        if 'training' in config_dict:
            self._update_config(self.training_config, config_dict['training'])
        if 'inference' in config_dict:
            self._update_config(self.inference_config, config_dict['inference'])
        if 'evaluation' in config_dict:
            self._update_config(self.evaluation_config, config_dict['evaluation'])
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        config_dict = {
            'model': asdict(self.model_config),
            'data': asdict(self.data_config),
            'training': asdict(self.training_config),
            'inference': asdict(self.inference_config),
            'evaluation': asdict(self.evaluation_config)
        }
        
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def _update_config(self, config_obj, config_dict: Dict[str, Any]):
        """更新配置对象"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model': asdict(self.model_config),
            'data': asdict(self.data_config),
            'training': asdict(self.training_config),
            'inference': asdict(self.inference_config),
            'evaluation': asdict(self.evaluation_config)
        } 