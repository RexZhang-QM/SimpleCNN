dependencies = ['torch']

# 定义加载 SimpleCNN 模型的函数
def simple_model(num_classes=10, **kwargs):
    """加载 SimpleCNN 模型."""
    model = SimpleCNN(num_classes=num_classes)
    return model
