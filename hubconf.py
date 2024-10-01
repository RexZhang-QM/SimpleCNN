import torch
import requests
from io import BytesIO
from SimpleCNN_model_for_hub import SimpleCNN

# 确保输出类别数与预训练权重匹配
num_classes = 10

model = SimpleCNN(num_classes=num_classes)
model.eval()

# 直接加载预训练模型权重
model_url = 'https://raw.githubusercontent.com/RexZhang-QM/SimpleCNN/master/model_trained.pth'
response = requests.get(model_url)
model_weights = BytesIO(response.content)
model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
