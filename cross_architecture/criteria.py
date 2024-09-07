import torch
from utils import set_seed

set_seed(42)
'''
ContrastLoss继承torch.nn.Module的原因
可组合性 继承 torch.nn.Module 使得 ContrastLoss 可以像其他 PyTorch 模块一样,轻松地集成到神经网络模型中。
它可以作为一个独立的模块,与其他模块(如卷积层、全连接层等)连接在一起,构建复杂的神经网络架构。
自动求导 torch.nn.Module 会自动初始化一个 backward 函数,该函数可以用来计算输出相对于参数的梯度。
通过继承它,ContrastLoss 可以利用 PyTorch 的自动微分机制,自动计算损失函数相对于输入的梯度,从而实现反向传播过程。
'''
# 自定义的CosineEmbeddingLoss
class PairwiseHingeLoss(torch.nn.Module):
    def __init__(self, margin=0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, score, target):
        # Compute the loss based on the target labels
        loss = torch.where(target == 1, 1 - score, 
                           torch.clamp(score - self.margin, min=0))
        
        # Apply the reduction method
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class PairwiseSoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores):
        # 假设scores形状为(batch_size, 2)
        # 第一列是正样本分数,第二列是负样本分数
        # 计算softmax概率分布
        prob_dist = scores.softmax(dim=1)
        
        # 提取正样本概率
        pos_prob = prob_dist[:, 0]
        
        # 计算负样本概率
        neg_prob = 1 - pos_prob
        
        # 计算负样本概率的均值作为损失
        loss = neg_prob.mean()
        
        return loss


# loss_function = PairwiseHingeLoss()
# # score = torch.Tensor([0.7, 0.3, 0.8, 0.6])
# score = torch.tensor([[0.7, 0.3], [0.8, 0.6]])

# # target = torch.Tensor([1, 0, 1, 0])
# # 使用 repeat 方法构建 target 张量，每行都是 [1, 0]
# target = torch.tensor([1, 0], dtype=torch.long).repeat(score.size()[0], 1)

# loss = loss_function(score, target)
# print(loss)

# 示例
batch_size = 3
num_classes = 2

# 创建一个维度为 [batch_size, num_classes] 的 scores 张量
scores = torch.randn(batch_size, num_classes)

# 创建一个维度为 [batch_size, num_classes] 的 target 张量，每行的数据都是 [1, 0]，类型为 torch.long
target = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.long)

# 创建损失函数对象
loss_fn = PairwiseHingeLoss(margin=0.5, reduction='mean')

# 计算二维张量的损失
loss_2d = loss_fn(scores, target)

# 将 scores 和 target 展开为一维张量
scores_flat = scores.view(-1)
target_flat = target.view(-1)

# 计算一维张量的损失
loss_1d = loss_fn(scores_flat, target_flat)

print("Scores (2D):")
print(scores)
print("Target (2D):")
print(target)
print("Loss (2D):")
print(loss_2d)

print("\nScores (1D):")
print(scores_flat)
print("Target (1D):")
print(target_flat)
print("Loss (1D):")
print(loss_1d)