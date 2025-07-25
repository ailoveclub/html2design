#!/usr/bin/env python3
"""
Qwen3 训练框架快速开始脚本
一键运行完整的训练流程演示
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src import Qwen3Trainer, Qwen3Inference, ModelEvaluator
from src.config import ConfigManager
from src.data import DataLoader
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_html_figma_data():
    """创建 HTML 转 Figma JSON 的示例数据"""
    logger.info("创建 HTML 转 Figma JSON 示例数据...")
    
    # 创建数据目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # HTML 转 Figma JSON 的示例数据
    html_figma_data = [
        {
            "input": '<div class="container"><h1>主标题</h1><p>这是一段描述文字。</p></div>',
            "output": '{"type": "FRAME", "name": "container", "children": [{"type": "TEXT", "characters": "主标题", "fontSize": 24, "fontWeight": 600}, {"type": "TEXT", "characters": "这是一段描述文字。", "fontSize": 16}]}'
        },
        {
            "input": '<button class="btn btn-primary">点击提交</button>',
            "output": '{"type": "FRAME", "name": "button", "children": [{"type": "TEXT", "characters": "点击提交", "fontSize": 14}], "fills": [{"type": "SOLID", "color": {"r": 0.0, "g": 0.5, "b": 1.0}}], "cornerRadius": 4}'
        },
        {
            "input": '<img src="avatar.jpg" alt="用户头像" width="100" height="100">',
            "output": '{"type": "RECTANGLE", "name": "avatar", "fills": [{"type": "IMAGE"}], "constraints": {"horizontal": "MIN", "vertical": "MIN"}, "effects": [{"type": "DROP_SHADOW"}]}'
        },
        {
            "input": '<nav><ul><li><a href="#home">首页</a></li><li><a href="#about">关于</a></li></ul></nav>',
            "output": '{"type": "FRAME", "name": "navigation", "layoutMode": "HORIZONTAL", "children": [{"type": "TEXT", "characters": "首页"}, {"type": "TEXT", "characters": "关于"}]}'
        },
        {
            "input": '<form><input type="email" placeholder="邮箱地址"><input type="password" placeholder="密码"></form>',
            "output": '{"type": "FRAME", "name": "form", "layoutMode": "VERTICAL", "children": [{"type": "RECTANGLE", "name": "email_input", "fills": [{"type": "SOLID", "color": {"r": 0.95, "g": 0.95, "b": 0.95}}]}, {"type": "RECTANGLE", "name": "password_input", "fills": [{"type": "SOLID", "color": {"r": 0.95, "g": 0.95, "b": 0.95}}]}]}'
        }
    ]
    
    # 扩展数据集
    import json
    train_data = html_figma_data * 20  # 重复数据用于演示
    val_data = html_figma_data[:2]
    test_data = html_figma_data[:3]
    
    # 保存训练数据
    train_file = data_dir / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证数据
    val_file = data_dir / "val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存测试数据
    test_file = data_dir / "test.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"数据已保存到: {data_dir}")
    return str(train_file), str(val_file), str(test_file)


def run_quick_demo():
    """运行快速演示"""
    logger.info("🚀 开始 Qwen3 训练框架快速演示")
    
    try:
        # 1. 创建示例数据
        train_file, val_file, test_file = create_html_figma_data()
        
        # 2. 加载配置
        logger.info("📋 加载配置...")
        config_manager = ConfigManager("configs/default_config.yaml")
        
        # 更新数据路径
        config_manager.data_config.train_file = train_file
        config_manager.data_config.val_file = val_file
        config_manager.data_config.test_file = test_file
        
        # 调整训练参数（用于快速演示）
        config_manager.training_config.num_train_epochs = 1
        config_manager.training_config.save_steps = 50
        config_manager.training_config.eval_steps = 50
        config_manager.training_config.logging_steps = 10
        
        logger.info(f"使用模型: {config_manager.model_config.model_name}")
        logger.info(f"训练轮数: {config_manager.training_config.num_train_epochs}")
        
        # 3. 创建训练器
        logger.info("🏗️ 创建训练器...")
        trainer = Qwen3Trainer(
            model_config=config_manager.model_config,
            training_config=config_manager.training_config,
            data_config=config_manager.data_config,
            use_lora=True
        )
        
        # 4. 开始训练（可选）
        user_input = input("\n是否开始训练？(y/n，默认跳过): ").strip().lower()
        if user_input == 'y':
            logger.info("🎯 开始训练...")
            result = trainer.run_full_training()
            logger.info(f"训练完成！损失: {result.metrics.get('train_loss', 'N/A')}")
        else:
            logger.info("⏭️ 跳过训练，使用预训练模型进行演示")
        
        # 5. 推理演示
        logger.info("🤖 开始推理演示...")
        
        from src.config import ModelConfig, InferenceConfig
        
        model_config = ModelConfig()
        inference_config = InferenceConfig()
        inference_config.max_new_tokens = 200
        inference_config.temperature = 0.7
        
        # 创建推理引擎（使用基础模型演示）
        inference_engine = Qwen3Inference(
            model_config=model_config,
            inference_config=inference_config
        )
        
        # 如果GPU内存足够，可以尝试加载模型
        try_inference = input("\n是否尝试推理演示？(y/n，需要GPU，默认跳过): ").strip().lower()
        if try_inference == 'y':
            try:
                logger.info("加载模型中...")
                inference_engine.load_model()
                
                # 进行推理演示
                html_examples = [
                    '<div class="card"><h2>产品标题</h2><p>产品描述</p><button>购买</button></div>',
                    '<header><nav><a href="#home">首页</a><a href="#about">关于我们</a></nav></header>'
                ]
                
                for i, html in enumerate(html_examples, 1):
                    logger.info(f"\n推理示例 {i}:")
                    logger.info(f"输入 HTML: {html}")
                    
                    response = inference_engine.generate(f"请将以下HTML转换为Figma JSON格式: {html}")
                    logger.info(f"生成的 Figma JSON: {response[:200]}...")
                    
            except Exception as e:
                logger.warning(f"推理演示跳过（GPU内存不足或其他问题）: {e}")
        
        # 6. 评估演示
        logger.info("📊 评估演示...")
        
        from src.config import EvaluationConfig
        eval_config = EvaluationConfig()
        evaluator = ModelEvaluator(eval_config)
        
        # 模拟评估数据
        predictions = [
            '{"type": "FRAME", "children": [{"type": "TEXT", "characters": "标题"}]}',
            '{"type": "RECTANGLE", "fills": [{"type": "SOLID"}]}',
            '{"type": "FRAME", "layoutMode": "HORIZONTAL"}'
        ]
        
        references = [
            '{"type": "FRAME", "children": [{"type": "TEXT", "characters": "主标题"}]}',
            '{"type": "RECTANGLE", "fills": [{"type": "SOLID", "color": {"r": 1}}]}',
            '{"type": "FRAME", "layoutMode": "VERTICAL"}'
        ]
        
        try:
            results = evaluator.evaluate_predictions(predictions, references)
            logger.info("评估结果:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        except Exception as e:
            logger.warning(f"评估演示跳过（缺少依赖）: {e}")
        
        # 7. 配置保存演示
        logger.info("💾 保存配置演示...")
        config_manager.save_to_file("demo_config.yaml")
        logger.info("配置已保存到 demo_config.yaml")
        
        logger.info("\n✅ 快速演示完成！")
        logger.info("📚 查看完整文档: README.md")
        logger.info("🔧 运行训练: qwen3-train --create-sample-data --use-lora")
        logger.info("🤖 运行推理: qwen3-infer --interactive")
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
        logger.info("请检查依赖是否安装完整，或查看README.md获取帮助")


def main():
    """主函数"""
    print("🎉 欢迎使用 Qwen3 训练框架！")
    print("这是一个用于微调 Qwen3 模型的完整训练框架")
    print("特别适用于 HTML 转 Figma JSON 等场景")
    print("-" * 50)
    
    # 检查基本依赖
    try:
        import torch
        import transformers
        logger.info(f"✅ PyTorch 版本: {torch.__version__}")
        logger.info(f"✅ Transformers 版本: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"❌ 缺少依赖: {e}")
        logger.info("请运行: pip install -r requirements.txt")
        return
    
    try:
        run_quick_demo()
    except KeyboardInterrupt:
        logger.info("\n👋 用户中断，再见！")
    except Exception as e:
        logger.error(f"运行出错: {e}")


if __name__ == "__main__":
    main() 