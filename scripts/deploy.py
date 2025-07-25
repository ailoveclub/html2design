#!/usr/bin/env python3
"""
Qwen3 模型部署脚本
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.deployment import DeploymentManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 模型部署")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["huggingface", "docker", "package"],
        required=True,
        help="部署动作"
    )
    
    # Hugging Face 相关参数
    parser.add_argument(
        "--repo-name",
        type=str,
        help="Hugging Face 仓库名称（格式：用户名/仓库名）"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="创建私有仓库"
    )
    parser.add_argument(
        "--model-description",
        type=str,
        default="基于 Qwen3 训练框架微调的模型",
        help="模型描述"
    )
    
    # 通用参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./deployment",
        help="输出目录"
    )
    parser.add_argument(
        "--include-docker",
        action="store_true",
        default=True,
        help="包含 Docker 文件"
    )
    parser.add_argument(
        "--include-service",
        action="store_true",
        default=True,
        help="包含推理服务"
    )
    parser.add_argument(
        "--gpu-support",
        action="store_true",
        help="GPU 支持"
    )
    
    args = parser.parse_args()
    
    try:
        # 检查模型路径
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 创建部署管理器
        hf_token = args.hf_token or os.getenv("HF_TOKEN")
        deployer = DeploymentManager(hf_token=hf_token)
        
        if args.action == "huggingface":
            # 部署到 Hugging Face Hub
            if not args.repo_name:
                raise ValueError("部署到 Hugging Face 需要指定 --repo-name")
            
            if not hf_token:
                raise ValueError("部署到 Hugging Face 需要提供 token（通过 --hf-token 或环境变量 HF_TOKEN）")
            
            logger.info(f"开始部署到 Hugging Face: {args.repo_name}")
            
            # 训练信息（可以根据实际情况修改）
            training_info = {
                "data": "自定义数据集",
                "procedure": "使用 Qwen3 训练框架进行 LoRA 微调",
                "usage_example": """
        from src import Qwen3Inference, ModelConfig, InferenceConfig

# 配置
model_config = ModelConfig()
inference_config = InferenceConfig()

# 创建推理引擎
inference = Qwen3Inference(model_config, inference_config, model_path="./model")
inference.load_model()

# 生成文本
response = inference.generate("你好")
print(response)
""",
                "limitations": "本模型基于特定数据集训练，可能存在领域偏见。请在使用前充分测试。"
            }
            
            # 部署
            url = deployer.deploy_to_huggingface(
                model_path=str(model_path),
                repo_name=args.repo_name,
                model_description=args.model_description,
                training_info=training_info,
                private=args.private
            )
            
            logger.info(f"模型已成功部署到: {url}")
        
        elif args.action == "docker":
            # 创建 Docker 部署文件
            logger.info(f"创建 Docker 部署文件: {args.output_dir}")
            
            deployer.create_deployment_package(
                model_path=str(model_path),
                output_dir=args.output_dir,
                include_docker=True,
                include_service=args.include_service,
                gpu_support=args.gpu_support
            )
            
            logger.info("Docker 部署文件创建完成")
            logger.info("使用以下命令构建和运行:")
            logger.info(f"  cd {args.output_dir}")
            logger.info("  docker build -t qwen3-service .")
            logger.info("  docker run -p 8000:8000 qwen3-service")
        
        elif args.action == "package":
            # 创建完整部署包
            logger.info(f"创建部署包: {args.output_dir}")
            
            deployer.create_deployment_package(
                model_path=str(model_path),
                output_dir=args.output_dir,
                include_docker=args.include_docker,
                include_service=args.include_service,
                gpu_support=args.gpu_support
            )
            
            logger.info("部署包创建完成")
            logger.info(f"查看部署说明: {Path(args.output_dir) / 'DEPLOYMENT.md'}")
        
    except Exception as e:
        logger.error(f"部署过程中出错: {e}")
        raise


if __name__ == "__main__":
    main() 