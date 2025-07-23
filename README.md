# Qwen3 训练框架

一个基于 Hugging Face 的 Qwen3 文本生成模型训练框架，支持数据准备、训练、评估、推理和部署等完整流程。

## 功能特性

- 🚀 **完整的训练流程**: 数据准备、模型训练、评估、推理、部署
- 🎯 **支持 LoRA 微调**: 高效的参数高效微调方法  
- ⚙️ **灵活的配置系统**: 支持 YAML/JSON 配置文件和命令行参数
- 📊 **丰富的评估指标**: BLEU、ROUGE、精确匹配等多种评估指标
- 🌐 **便捷的部署选项**: 支持 Hugging Face Hub、Docker 等部署方式
- 💻 **友好的命令行工具**: 提供训练、推理、评估、部署等命令行工具

## 安装

### 从源码安装

```bash
git clone https://github.com/yourusername/qwen3-trainer.git
cd qwen3-trainer
pip install -e .
```

### 依赖要求

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- 其他依赖见 `requirements.txt`

## 快速开始

### 1. 准备数据

创建训练数据文件（JSONL 格式）：

```json
{"input": "什么是人工智能？", "output": "人工智能是一门研究如何让机器模拟人类智能的学科。"}
{"input": "机器学习有哪些类型？", "output": "机器学习主要包括监督学习、无监督学习和强化学习三种类型。"}
```

### 2. 训练模型

使用默认配置训练：

```bash
qwen3-train --create-sample-data --use-lora
```

使用自定义数据训练：

```bash
qwen3-train --train-file data/train.jsonl --val-file data/val.jsonl --use-lora
```

### 3. 推理测试

交互式推理：

```bash
qwen3-infer --model-path ./outputs --is-lora --interactive
```

单次推理：

```bash
qwen3-infer --model-path ./outputs --is-lora --prompt "什么是深度学习？"
```

启动推理服务：

```bash
qwen3-infer --model-path ./outputs --is-lora --start-server
```

### 4. 模型评估

```bash
qwen3-eval --model-path ./outputs --test-file data/test.jsonl --is-lora --generate-report
```

### 5. 模型部署

部署到 Hugging Face Hub：

```bash
qwen3-deploy --model-path ./outputs --action huggingface --repo-name yourusername/your-model
```

创建 Docker 部署包：

```bash
qwen3-deploy --model-path ./outputs --action docker --output-dir ./deployment
```

## 详细使用说明

### 配置文件

框架支持 YAML 和 JSON 格式的配置文件。默认配置文件位于 `configs/default_config.yaml`：

```yaml
# 模型配置
model:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "bfloat16"
  use_flash_attention: true

# 训练配置  
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 5.0e-5
  
# LoRA 配置
lora:
  use_lora: true
  r: 16
  lora_alpha: 32
```

### Python API 使用

#### 训练模型

```python
from qwen3_trainer import Qwen3Trainer
from qwen3_trainer.config import ConfigManager

# 加载配置
config = ConfigManager("configs/default_config.yaml")

# 创建训练器
trainer = Qwen3Trainer(
    model_config=config.model_config,
    training_config=config.training_config,
    data_config=config.data_config,
    use_lora=True
)

# 开始训练
trainer.run_full_training()
```

#### 模型推理

```python
from qwen3_trainer import Qwen3Inference, ModelConfig, InferenceConfig

# 配置
model_config = ModelConfig()
inference_config = InferenceConfig()

# 创建推理引擎
inference = Qwen3Inference(
    model_config=model_config,
    inference_config=inference_config,
    model_path="./outputs",
    is_lora_model=True
)

# 加载模型
inference.load_model()

# 生成文本
response = inference.generate("什么是机器学习？")
print(response)
```

#### 模型评估

```python
from qwen3_trainer import ModelEvaluator, EvaluationConfig

# 配置评估
eval_config = EvaluationConfig()
eval_config.metrics = ["bleu", "rouge", "exact_match"]

# 创建评估器
evaluator = ModelEvaluator(eval_config)

# 评估模型
results = evaluator.evaluate_model_on_dataset(
    model=inference,
    test_data=test_data
)

print(f"BLEU 分数: {results['bleu']:.4f}")
```

### 数据格式

框架支持多种数据格式：

#### JSONL 格式（推荐）

```json
{"input": "输入文本1", "output": "输出文本1"}
{"input": "输入文本2", "output": "输出文本2"}
```

#### JSON 格式

```json
[
  {"input": "输入文本1", "output": "输出文本1"},
  {"input": "输入文本2", "output": "输出文本2"}
]
```

#### CSV 格式

```csv
input,output
输入文本1,输出文本1
输入文本2,输出文本2
```

### 自定义评估指标

框架支持添加自定义评估指标：

```python
from qwen3_trainer.evaluator import CustomMetrics

# HTML 结构相似度（适用于 HTML 转换任务）
similarity = CustomMetrics.html_structure_similarity(pred_html, ref_html)

# JSON 键覆盖率（适用于 JSON 生成任务）
coverage = CustomMetrics.json_key_coverage(pred_json, ref_json)
```

## 适用场景

本框架特别适用于以下场景：

1. **HTML 转 Figma JSON**: 将 HTML 代码转换为 Figma 设计文件格式
2. **代码生成**: 根据自然语言描述生成代码
3. **文本摘要**: 长文本的自动摘要生成
4. **对话系统**: 构建聊天机器人和对话 AI
5. **内容创作**: 自动化文章、报告等内容生成

## 目录结构

```
qwen3-trainer/
├── qwen3_trainer/          # 核心框架代码
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── data.py             # 数据处理
│   ├── trainer.py          # 训练模块
│   ├── inference.py        # 推理模块
│   ├── evaluator.py        # 评估模块
│   └── deployment.py       # 部署模块
├── scripts/                # 命令行脚本
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   ├── evaluate.py        # 评估脚本
│   └── deploy.py          # 部署脚本
├── configs/               # 配置文件
│   └── default_config.yaml
├── examples/              # 使用示例
├── requirements.txt       # 依赖列表
├── setup.py              # 安装脚本
└── README.md             # 说明文档
```

## 性能优化建议

1. **使用 GPU**: 训练和推理时建议使用 GPU 加速
2. **调整批大小**: 根据显存大小调整 batch_size
3. **梯度累积**: 在显存不足时使用 gradient_accumulation_steps
4. **混合精度**: 启用 bf16 或 fp16 减少显存使用
5. **LoRA 微调**: 对于大模型推荐使用 LoRA 微调

## 故障排除

### 常见问题

1. **显存不足**
   ```bash
   # 减小批大小
   --batch-size 1
   # 启用梯度检查点
   --gradient-checkpointing
   ```

2. **模型加载失败**
   ```bash
   # 检查模型路径
   ls ./outputs
   # 确认模型文件完整性
   ```

3. **推理速度慢**
   ```bash
   # 检查是否使用 GPU
   nvidia-smi
   # 启用 Flash Attention
   pip install flash-attn
   ```

## 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -am 'Add some feature'`
4. 推送分支: `git push origin feature/your-feature`
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 更新日志

### v0.1.0 (2024-01-xx)

- 初始版本发布
- 支持 Qwen3 模型微调
- 实现完整的训练、推理、评估流程
- 提供命令行工具和 Python API
- 支持 Hugging Face Hub 和 Docker 部署

## 联系我们

- 项目主页: https://github.com/yourusername/qwen3-trainer
- 问题反馈: https://github.com/yourusername/qwen3-trainer/issues
- 邮箱: your.email@example.com

## 致谢

感谢以下项目和组织：

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [Qwen](https://github.com/QwenLM/Qwen)
- [PyTorch](https://pytorch.org/) 