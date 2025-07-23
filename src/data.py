"""
数据处理模块
包含数据加载、预处理、分词等功能
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import logging

from .config import DataConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_data(self, file_path: str) -> Dataset:
        """加载数据文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        if self.config.data_format == "jsonl":
            data = self._load_jsonl(file_path)
        elif self.config.data_format == "json":
            data = self._load_json(file_path)
        elif self.config.data_format == "csv":
            data = self._load_csv(file_path)
        else:
            raise ValueError(f"不支持的数据格式: {self.config.data_format}")
        
        return Dataset.from_list(data)
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行: {line}, 错误: {e}")
        return data
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("JSON文件格式不正确，应该是字典或字典列表")
    
    def _load_csv(self, file_path: Path) -> List[Dict]:
        """加载CSV文件"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """数据预处理函数"""
        inputs = examples[self.config.input_column]
        outputs = examples[self.config.output_column]
        
        # 构建输入文本
        formatted_inputs = []
        for inp, out in zip(inputs, outputs):
            # 使用Qwen的对话格式
            formatted_input = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            formatted_inputs.append(formatted_input)
        
        # 分词
        model_inputs = self.tokenizer(
            formatted_inputs,
            max_length=self.config.max_input_length + self.config.max_output_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # 设置labels（用于计算loss）
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """准备训练数据集"""
        # 检查必要的列是否存在
        if self.config.input_column not in dataset.column_names:
            raise ValueError(f"数据集中缺少输入列: {self.config.input_column}")
        if self.config.output_column not in dataset.column_names:
            raise ValueError(f"数据集中缺少输出列: {self.config.output_column}")
        
        # 应用预处理
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc="预处理数据"
        )
        
        return processed_dataset


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.processor = DataProcessor(config, tokenizer)
    
    def load_train_dataset(self) -> Optional[Dataset]:
        """加载训练数据集"""
        if not self.config.train_file:
            return None
        
        logger.info(f"加载训练数据: {self.config.train_file}")
        dataset = self.processor.load_data(self.config.train_file)
        return self.processor.prepare_dataset(dataset)
    
    def load_val_dataset(self) -> Optional[Dataset]:
        """加载验证数据集"""
        if not self.config.val_file:
            return None
        
        logger.info(f"加载验证数据: {self.config.val_file}")
        dataset = self.processor.load_data(self.config.val_file)
        return self.processor.prepare_dataset(dataset)
    
    def load_test_dataset(self) -> Optional[Dataset]:
        """加载测试数据集"""
        if not self.config.test_file:
            return None
        
        logger.info(f"加载测试数据: {self.config.test_file}")
        dataset = self.processor.load_data(self.config.test_file)
        return self.processor.prepare_dataset(dataset)
    
    def create_sample_data(self, output_dir: str = "data", num_samples: int = 100):
        """创建示例数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建示例训练数据
        train_data = []
        for i in range(num_samples):
            train_data.append({
                self.config.input_column: f"这是第{i+1}个示例输入",
                self.config.output_column: f"这是第{i+1}个示例输出"
            })
        
        # 保存训练数据
        train_file = output_path / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 创建示例验证数据
        val_data = []
        for i in range(num_samples // 5):
            val_data.append({
                self.config.input_column: f"这是第{i+1}个验证输入",
                self.config.output_column: f"这是第{i+1}个验证输出"
            })
        
        # 保存验证数据
        val_file = output_path / "val.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"示例数据已保存到: {output_path}")
        return str(train_file), str(val_file)


def collate_fn(examples: List[Dict]) -> Dict:
    """数据整理函数，用于DataLoader"""
    batch = {}
    
    # 获取所有键
    keys = examples[0].keys()
    
    for key in keys:
        values = [example[key] for example in examples]
        if key in ['input_ids', 'attention_mask', 'labels']:
            # 对序列进行padding
            max_length = max(len(v) for v in values)
            padded_values = []
            
            for value in values:
                if key == 'labels':
                    # labels用-100填充
                    padded_value = value + [-100] * (max_length - len(value))
                else:
                    # input_ids和attention_mask用0填充
                    padded_value = value + [0] * (max_length - len(value))
                padded_values.append(padded_value)
            
            batch[key] = padded_values
        else:
            batch[key] = values
    
    return batch 