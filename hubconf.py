dependencies = ['torch', 'requests']

import requests
import torch
from SimpleCNN_model_for_hub import SimpleCNN

def simple_model(num_classes=10, **kwargs):
    """直接加载预训练的 SimpleCNN 模型权重。

    Args:
    - num_classes (int): 输出类别数。
    """
    print("直接加载模型权重...")
    model_url = 'https://raw.githubusercontent.com/RexZhang-QM/SimpleCNN/master/model_trained.pth'
    model_weights = requests.get(model_url).content
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    model.eval()
    return model
