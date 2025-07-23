"""
部署模块
支持模型发布到 Hugging Face Hub 和其他部署选项
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import shutil

try:
    from huggingface_hub import HfApi, Repository, create_repo, upload_folder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from .config import ModelConfig, InferenceConfig

logger = logging.getLogger(__name__)


class HuggingFaceDeployer:
    """Hugging Face Hub 部署器"""
    
    def __init__(self, token: Optional[str] = None):
        if not HF_HUB_AVAILABLE:
            raise ImportError("需要安装 huggingface_hub: pip install huggingface_hub")
        
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            logger.warning("未提供 Hugging Face token，某些功能可能不可用")
        
        self.api = HfApi(token=self.token)
    
    def upload_model(
        self,
        model_path: str,
        repo_name: str,
        private: bool = False,
        commit_message: str = "Upload model",
        create_if_not_exists: bool = True
    ) -> str:
        """上传模型到 Hugging Face Hub"""
        if not self.token:
            raise ValueError("需要提供 Hugging Face token")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        try:
            # 创建仓库（如果不存在）
            if create_if_not_exists:
                try:
                    create_repo(
                        repo_id=repo_name,
                        token=self.token,
                        private=private,
                        exist_ok=True
                    )
                    logger.info(f"仓库已创建或已存在: {repo_name}")
                except Exception as e:
                    logger.warning(f"创建仓库失败: {e}")
            
            # 上传模型文件
            url = upload_folder(
                folder_path=str(model_path),
                repo_id=repo_name,
                token=self.token,
                commit_message=commit_message
            )
            
            logger.info(f"模型已成功上传到: {url}")
            return url
            
        except Exception as e:
            logger.error(f"上传模型失败: {e}")
            raise
    
    def create_model_card(
        self,
        repo_name: str,
        model_description: str,
        training_data: str,
        training_procedure: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        usage_example: Optional[str] = None,
        limitations: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建模型卡片（README.md）"""
        
        model_card = f"""# {repo_name}

## 模型描述

{model_description}

## 训练数据

{training_data}

## 训练过程

{training_procedure}

"""
        
        # 添加性能指标
        if performance_metrics:
            model_card += "## 性能指标\n\n"
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    model_card += f"- {metric}: {value:.4f}\n"
                else:
                    model_card += f"- {metric}: {value}\n"
            model_card += "\n"
        
        # 添加使用示例
        if usage_example:
            model_card += f"""## 使用示例

```python
{usage_example}
```

"""
        
        # 添加限制说明
        if limitations:
            model_card += f"""## 限制和偏见

{limitations}

"""
        
        # 添加其他信息
        if additional_info:
            model_card += "## 其他信息\n\n"
            for key, value in additional_info.items():
                model_card += f"- {key}: {value}\n"
        
        # 添加框架信息
        model_card += """
## 框架信息

本模型使用 Qwen3 训练框架进行训练，基于 Hugging Face Transformers 和 PEFT。

### 依赖项

```bash
pip install torch transformers datasets peft
```

### 推理代码

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("你的用户名/仓库名")
model = AutoModelForCausalLM.from_pretrained("你的用户名/仓库名")

# 生成文本
input_text = "你的输入"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

"""
        
        return model_card
    
    def save_model_card(self, model_card: str, output_path: str):
        """保存模型卡片到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        logger.info(f"模型卡片已保存到: {output_path}")
    
    def prepare_for_upload(
        self,
        model_path: str,
        output_dir: str,
        model_card: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """准备模型用于上传"""
        model_path = Path(model_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        for file_pattern in ["*.bin", "*.safetensors", "*.json", "*.txt"]:
            for file_path in model_path.glob(file_pattern):
                shutil.copy2(file_path, output_path / file_path.name)
        
        # 保存模型卡片
        if model_card:
            readme_path = output_path / "README.md"
            self.save_model_card(model_card, str(readme_path))
        
        # 更新配置文件（如果有覆盖设置）
        if config_overrides:
            config_path = output_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                config.update(config_overrides)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已准备完成: {output_path}")
        return str(output_path)


class DockerDeployer:
    """Docker 部署器"""
    
    def __init__(self):
        pass
    
    def create_dockerfile(
        self,
        base_image: str = "python:3.9-slim",
        model_path: str = "./model",
        port: int = 8000,
        requirements_file: str = "requirements.txt"
    ) -> str:
        """创建 Dockerfile"""
        
        dockerfile_content = f"""# 使用官方 Python 镜像
FROM {base_image}

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY {requirements_file} .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r {requirements_file}

# 复制模型和代码
COPY {model_path} ./model/
COPY *.py ./
COPY qwen3_trainer/ ./qwen3_trainer/

# 暴露端口
EXPOSE {port}

# 设置环境变量
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model

# 启动命令
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "{port}"]
"""
        
        return dockerfile_content
    
    def create_docker_compose(
        self,
        service_name: str = "qwen3-service",
        port: int = 8000,
        gpu_support: bool = False
    ) -> str:
        """创建 docker-compose.yml"""
        
        compose_content = f"""version: '3.8'

services:
  {service_name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - MODEL_PATH=/app/model
    volumes:
      - ./model:/app/model:ro
"""
        
        if gpu_support:
            compose_content += """    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
        
        return compose_content
    
    def save_docker_files(
        self,
        output_dir: str,
        dockerfile_content: str,
        compose_content: str
    ):
        """保存 Docker 相关文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 Dockerfile
        dockerfile_path = output_path / "Dockerfile"
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        # 保存 docker-compose.yml
        compose_path = output_path / "docker-compose.yml"
        with open(compose_path, 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        logger.info(f"Docker 文件已保存到: {output_path}")


class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_deployer = None
        self.docker_deployer = DockerDeployer()
        
        if HF_HUB_AVAILABLE:
            self.hf_deployer = HuggingFaceDeployer(hf_token)
    
    def deploy_to_huggingface(
        self,
        model_path: str,
        repo_name: str,
        model_description: str,
        training_info: Dict[str, str],
        performance_metrics: Optional[Dict[str, Any]] = None,
        private: bool = False
    ) -> str:
        """部署模型到 Hugging Face Hub"""
        if not self.hf_deployer:
            raise ImportError("Hugging Face Hub 不可用")
        
        # 创建模型卡片
        model_card = self.hf_deployer.create_model_card(
            repo_name=repo_name,
            model_description=model_description,
            training_data=training_info.get("data", "未提供"),
            training_procedure=training_info.get("procedure", "未提供"),
            performance_metrics=performance_metrics,
            usage_example=training_info.get("usage_example"),
            limitations=training_info.get("limitations")
        )
        
        # 准备上传
        temp_dir = f"./temp_upload_{repo_name.replace('/', '_')}"
        prepared_path = self.hf_deployer.prepare_for_upload(
            model_path=model_path,
            output_dir=temp_dir,
            model_card=model_card
        )
        
        # 上传模型
        url = self.hf_deployer.upload_model(
            model_path=prepared_path,
            repo_name=repo_name,
            private=private
        )
        
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return url
    
    def create_deployment_package(
        self,
        model_path: str,
        output_dir: str,
        include_docker: bool = True,
        include_service: bool = True,
        gpu_support: bool = False
    ):
        """创建完整的部署包"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        model_output = output_path / "model"
        if Path(model_path).exists():
            shutil.copytree(model_path, model_output, dirs_exist_ok=True)
        
        # 创建服务脚本
        if include_service:
            service_script = self._create_service_script()
            with open(output_path / "serve.py", 'w', encoding='utf-8') as f:
                f.write(service_script)
        
        # 创建 Docker 文件
        if include_docker:
            dockerfile = self.docker_deployer.create_dockerfile(
                gpu_support=gpu_support
            )
            compose = self.docker_deployer.create_docker_compose(
                gpu_support=gpu_support
            )
            
            self.docker_deployer.save_docker_files(
                output_dir=output_dir,
                dockerfile_content=dockerfile,
                compose_content=compose
            )
        
        # 创建部署说明
        readme = self._create_deployment_readme(include_docker, include_service)
        with open(output_path / "DEPLOYMENT.md", 'w', encoding='utf-8') as f:
            f.write(readme)
        
        logger.info(f"部署包已创建: {output_path}")
    
    def _create_service_script(self) -> str:
        """创建服务脚本"""
        return '''#!/usr/bin/env python3
"""
Qwen3 推理服务
"""

import argparse
import logging
from qwen3_trainer import Qwen3Inference, ModelConfig, InferenceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Qwen3 推理服务")
    parser.add_argument("--model-path", default="./model", help="模型路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--is-lora", action="store_true", help="是否为LoRA模型")
    
    args = parser.parse_args()
    
    # 配置模型
    model_config = ModelConfig()
    inference_config = InferenceConfig()
    
    # 创建推理引擎
    inference_engine = Qwen3Inference(
        model_config=model_config,
        inference_config=inference_config,
        model_path=args.model_path,
        is_lora_model=args.is_lora
    )
    
    # 加载模型
    logger.info("加载模型...")
    inference_engine.load_model()
    
    # 启动服务
    from qwen3_trainer.inference import InferenceServer
    server = InferenceServer(inference_engine)
    server.start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
'''
    
    def _create_deployment_readme(self, include_docker: bool, include_service: bool) -> str:
        """创建部署说明"""
        readme = """# 模型部署指南

本文档介绍如何部署训练好的 Qwen3 模型。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

"""
        
        if include_service:
            readme += """### 2. 启动推理服务

```bash
python serve.py --model-path ./model --host 0.0.0.0 --port 8000
```

### 3. 测试服务

```bash
curl -X POST "http://localhost:8000/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "你好，请介绍一下自己"}'
```

"""
        
        if include_docker:
            readme += """## Docker 部署

### 构建镜像

```bash
docker build -t qwen3-service .
```

### 运行容器

```bash
docker run -p 8000:8000 qwen3-service
```

### 使用 Docker Compose

```bash
docker-compose up -d
```

"""
        
        readme += """## API 文档

服务提供以下端点：

- `GET /health` - 健康检查
- `POST /generate` - 文本生成

### 生成文本

请求格式：
```json
{
  "prompt": "你的输入文本",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

响应格式：
```json
{
  "generated_text": "生成的文本",
  "prompt": "原始输入"
}
```

## 性能调优

1. **GPU 加速**: 确保安装了正确的 CUDA 版本
2. **批处理**: 对于高并发场景，考虑使用批处理
3. **模型量化**: 可以使用 8-bit 或 4-bit 量化减少内存使用

## 故障排除

常见问题和解决方案：

1. **内存不足**: 减小批大小或使用模型量化
2. **加载速度慢**: 使用本地模型缓存
3. **推理速度慢**: 检查 GPU 使用情况

"""
        
        return readme 