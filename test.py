import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModule, self).__init__()
        # 定义一个可学习的权重参数
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        # 定义一个可学习的偏置参数
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        return x @ self.weights + self.bias

# 创建模型实例
model = MyModule(input_size=5, output_size=3)

# 打印模型参数
for name, param in model.named_parameters():
    print(name, param)