
# MOE介绍及代码： https://blog.csdn.net/weixin_44986037/article/details/150105895

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_npu


class BasicExpert(nn.Module):
    """
    基础专家网络

    输入:
        x : 输入的特征向量(batch, feature_in)
    
    参数:
        feature_in : 输入特征向量的维度
        feature_out : 输出特征向量的维度, 也是Linear的嵌入维度emd_dim
    
    输出:
        单个专家处理后的输出向量
    """
    # 一个 Expert 可以是一个最简单的， linear 层即可
    # 也可以是 MLP 层
    # 也可以是 更复杂的 MLP 层（active function 设置为 swiglu）
    def __init__(self, feature_in, feature_out):
        super().__init__()
        # self.linear = nn.Linear(feature_in, feature_out)
        # 这里使用典型的 MLP 结构：Linear -> ReLU -> Linear
        self.net = nn.Sequential(
            nn.Linear(feature_in, feature_in * 2),
            nn.ReLU(),
            nn.Linear(feature_in * 2, feature_out)
        )
    
    def forward(self, x):
        # return self.linear(x)
        return self.net(x)
    



class BasicMOE(nn.Module):
    """
    MOE混合专家模型

    输入:
        x : 输入的特征向量(batch, feature_in)
    
    参数:
        feature_in : 输入特征向量的维度
        feature_out : 输出特征向量的维度, 也是Linear的嵌入维度emd_dim
        expert_number : 专家个数
    
    输出:
        output

    """
    def __init__(self, feature_in, feature_out, expert_number):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                BasicExpert(feature_in, feature_out) for _ in range(expert_number)
            ]
        )

        self.expert_number = expert_number

        # 专家路由【疑问：隐藏层如何选取?】
        # gate 就是选一个 expert 
        # self.gate = nn.Linear(feature_in, expert_number)
        self.gate = nn.Sequential(
            nn.Linear(feature_in, feature_in * 2),
            nn.ReLU(),
            nn.Linear(feature_in * 2, expert_number)
        )
    
    def forward(self, x):  
        "输入: 融合特征[2,501,100]  (batch_size, token_num, token_dim)"

        # A. 计算门控权重
        expert_weight = self.gate(x)  # shape 是 (batch, token_num, expert_number) (2，501, 3)
        expert_weight = F.softmax(expert_weight, dim=-1) # (batch, expert_number)
        
        # --- 计算负载均衡损失 (Auxiliary Loss) ---
        # f: 每个专家获得的权重均值 (Importance), 每个token被哪个专家选中
        f = expert_weight.mean(0)  # [501, 3] 
        # P: 每个专家被选中的概率均值 (实际上在 Dense MoE 中 P = f)
        # 在 Sparse MoE 中，P 通常是样本被分配到该专家的频率
        # 这里为了演示通用公式：Loss = N * sum(f_i * P_i)
        moe_loss = self.expert_number * torch.sum(f * f) # 每个专家被分配的概率


        # B. 获得所有专家输出
        expert_out_list = [
            expert(x).unsqueeze(2) for expert in self.experts
        ]  # unsequeeze(1)后,变成(batch, token_num, 1, feature_out) [2,501,1,256]*3

        # 拼接所有专家输出 : concat 起来 (batch, token_num, expert_number, feature_out)   (2,2,3)
        expert_output = torch.cat(expert_out_list, dim=2) #[2,501,3,256]
        # print(expert_output.size())

        # C. 加权求和
        expert_weight = expert_weight.unsqueeze(-1) # (batch, token_num, expert_num, 1)   (2,501,3,1,)

        # expert_weight * expert_out_list   (2,501,3,1) * (2, 501, 3, 256)
        output = torch.sum(expert_weight * expert_output, dim=2) # (batch, token_num, feature_out)
        # output = (expert_weight @ expert_output).squeeze()  # (batch, 1, feature_out)   (2,1,3)


        return output, moe_loss



"""
def test_basic_moe():
    x = torch.rand(2, 4)

    basic_moe = BasicMOE(4, 3, 2)
    out = basic_moe(x)
    print(out)
    print("out.shape:", out.shape)  #(batch, feature_out), from (2, 4) to (2,3) 


test_basic_moe()

"""