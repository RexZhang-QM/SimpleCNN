dependencies = ['torch']

from SimpleCNN_model_for_hub import SimpleCNN  # 确保你正确导入 SimpleCNN 类

def simple_model(num_classes=10, **kwargs):
    """加载 SimpleCNN 模型."""
    model = SimpleCNN(num_classes=num_classes)
    return model
