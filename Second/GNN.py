import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid          # 内置数据集
from torch_geometric.nn import GCNConv                  # 图卷积层
from torch_geometric.data import Data                   # 图数据容器
from torch_geometric.loader import DataLoader           # 数据加载器（批处理）
import torch_geometric.transforms as T                  # 数据变换工具

#加载 Cora 数据集
def load_cora_dataset():
    # 加载 Cora 数据
    # RandomNodeSplit 会将节点随机划分
    dataset = Planetoid(
        root='./data/Cora',  # 数据保存目录
        name='Cora',  # 数据集名称
        transform=T.Compose([
            T.NormalizeFeatures(),  # 归一化节点特征
            T.RandomNodeSplit(
                num_train_per_class=20,  # 每类20个训练节点
                num_val=500,  # 500个验证节点
                num_test=1000  # 1000个测试节点
            )
        ])
    )

    print(f"数据集: {dataset}")
    print(f"{dataset.num_edge_features}")
    print(f"类别数: {dataset.num_classes}")
    print(f"节点特征维度: {dataset.num_node_features}")
    print(f"图数量: {len(dataset)}")

    data_cora = dataset[0]
    print(f"\n图数据信息:")
    print(f"  - 节点数: {data_cora.num_nodes}")
    print(f"  - 边数: {data_cora.num_edges}")
    print(f"  - 训练节点数: {data_cora.train_mask.sum().item()}")
    print(f"  - 验证节点数: {data_cora.val_mask.sum().item()}")
    print(f"  - 测试节点数: {data_cora.test_mask.sum().item()}")

    return dataset, data_cora
#print(load_cora_dataset())



class GCN(torch.nn.Module):
    """
    两层图卷积网络（GCN）用于节点分类

    架构：
       输入 (in_channels) -> GCNConv -> ReLU -> Dropout -> GCNConv -> LogSoftmax -> 输出 (out_channels)
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        """
        初始化 GCN 模型
        Args:
            in_channels: 输入特征维度（节点特征数）
            hidden_channels: 隐藏层维度
            out_channels: 输出维度（类别数）
            dropout: Dropout 比率，用于防止过拟合
        """
        super(GCN, self).__init__()
        # 第一层图卷积：输入 > 隐藏层
        # GCNConv 是图卷积层，核心操作：聚合邻居信息 + 线性变换
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 第二层图卷积：隐藏层 > 输出层
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # Dropout 层，用于训练时随机丢弃部分神经元，防止过拟合
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 节点特征矩阵 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
        Returns:
            概率 [num_nodes, out_channels]
        """
        # ----- 第一层 -----
        # 图卷积：聚合邻居信息并做线性变换
        x = self.conv1(x, edge_index)
        # ReLU 激活函数：引入非线性
        x = F.relu(x)
        # Dropout：随机丢弃部分神经元
        x = F.dropout(x, p=self.dropout, training=self.training)
        # ----- 第二层 -----
        # 图卷积：输出 logits
        x = self.conv2(x, edge_index)
        # LogSoftmax：将输出转换为对数概率分布
        # dim=1 表示在类别维度上做 softmax
        return F.log_softmax(x, dim=1)

def train(model_train: torch.nn.Module, data_train: Data, optimizer: torch.optim.Optimizer,epoch: int) -> float:
    """
    训练一个 epoch
    Args:
        model_train: GNN 模型
        data_train: 图数据
        optimizer: 优化器
        epoch: 当前 epoch 编号
    Returns:
        当前 epoch 的训练损失
    """
    # 设置为训练模式
    model_train.train()
    # 梯度清零
    optimizer.zero_grad()
    # 前向传播：输入节点特征和边索引，得到预测结果
    out = model_train(data_train.x, data_train.edge_index)
    # 计算损失：使用负对数似然损失（NLLLoss）
    # 只计算训练集节点的损失（通过 train_mask 筛选）
    loss = F.nll_loss(out[data_train.train_mask], data_train.y[data_train.train_mask])
    # 反向传播：计算梯度
    loss.backward()
    # 更新参数
    optimizer.step()
    return loss.item()


def evaluate(model_evaluate: torch.nn.Module, data_evaluate: Data, mask: torch.Tensor) -> float:
    """
    评估模型在指定数据集上的准确率
    Args:
        model_evaluate: GNN 模型
        data_evaluate: 图数据
        mask: 节点掩码（train_mask / val_mask / test_mask）
    Returns:
        准确率（0~1 之间的浮点数）
    """
    # 设置为评估模式（禁用 Dropout）
    model_evaluate.eval()
    # 禁用梯度计算（节省内存和计算）
    with torch.no_grad():
        # 前向传播
        out = model_evaluate(data_evaluate.x, data_evaluate.edge_index)
        # 获取预测类别（取概率最大的类别）
        pred = out.argmax(dim=1)
        # 计算正确预测的数量
        correct = (pred[mask] == data_evaluate.y[mask]).sum().item()
        # 计算准确率
        acc = correct / mask.sum().item()
    return acc


def main():
    """
    数据加载 -> 模型创建 -> 训练 -> 评估
    """
    dataset, data = load_cora_dataset()
    in_channels = dataset.num_node_features                 # 输入特征维度：1433
    hidden_channels = 16                                    # 隐藏层维度
    out_channels = dataset.num_classes                      # 输出类别数：7
    dropout = 0.5                                           # Dropout 比率
    print(f"\n模型参数:")
    print(f"  - 输入维度: {in_channels}")
    print(f"  - 隐藏维度: {hidden_channels}")
    print(f"  - 输出维度: {out_channels}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels, hidden_channels, out_channels, dropout).to(device)
    data = data.to(device)
    print(f"\n使用设备: {device}")
    print(f"模型结构:\n{model}")
    # 使用 Adam，学习率 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    epochs = 200  # 训练轮数
    best_val_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(1, epochs + 1):
        # 训练一个 epoch
        loss = train(model, data, optimizer, epoch)
        # 每 10 个 epoch 评估一次
        if epoch % 10 == 0:
            # 计算训练集、验证集、测试集的准确率
            train_acc = evaluate(model, data, data.train_mask)
            val_acc = evaluate(model, data, data.val_mask)
            test_acc = evaluate(model, data, data.test_mask)
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, 'f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"对应测试准确率: {best_test_acc:.4f}")
    # 最终在测试集上评估
    final_test_acc = evaluate(model, data, data.test_mask)
    print(f"最终测试准确率: {final_test_acc:.4f}")
    return model, data

if __name__ == "__main__":
    #test：
    print(torch.__version__)  # 确定pytorch的版本
    #create_custom_graph()
    #load_cora_dataset()

    # 设置随机种子，确保结果可复现

    torch.manual_seed(42)

    # 运行主程序
   # model, data = main()