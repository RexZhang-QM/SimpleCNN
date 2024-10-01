dependencies = ['torch', 'requests']

import requests
import torch
from SimpleCNN_model_for_hub import SimpleCNN

def simple_model(num_classes=10, pretrained=False, **kwargs):
    """加载 SimpleCNN 模型或加载预训练的模型权重。

    Args:
    - num_classes (int): 输出类别数。
    - pretrained (bool): 是否加载预训练的模型权重。
    """
    #if pretrained:
        # 如果需要加载预训练的模型权重
        print("直接加载模型权重...")
        model_url = 'https://raw.githubusercontent.com/RexZhang-QM/SimpleCNN/model_trained.pth'
        model_weights = requests.get(model_url).content
        model = SimpleCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
        model.eval()
    #else:
        # 如果不需要加载预训练的模型权重，即创建一个新的模型实例
        print("不加载预训练的模型权重，创建一个新的模型实例...")
        model = SimpleCNN(num_classes=num_classes)

    return model
