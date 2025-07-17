# -----#
# author:HD
# year&month&day:2025:04:19
# -----#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from utils.SCN_dataread import SCN_dataread
from utils.SCN_metrics import PECN_metrics
from network.SCN import SCN
# 是否使用GPU
use_gpu = True
if use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

torch.cuda.empty_cache()
# 固定的训练参数

Epoch = 500
Batch = 512
model_save_path = './model/NLNet.pth'

# 数据路径
file_path = 'datasets/train_datasets/'
data_name_train = 'NLNet_train'
data_name_val = 'NLNet_val'
num = 4096

# 读取训练数据
train_data = SCN_dataread(file_path, data_name_train, label_num=num)
a, b = train_data.data_read(file_path, data_name_train, label_num=num)

train_loader = DataLoader(dataset=train_data,
                          batch_size=Batch,
                          shuffle=True,
                          pin_memory=False,
                          drop_last=True)

# 读取验证数据
val_data = SCN_dataread(file_path, data_name_val, label_num=num)
val_loader = DataLoader(dataset=val_data,
                        batch_size=1024,
                        shuffle=True,
                        pin_memory=False,
                        drop_last=True)

# 网格搜索参数：L和S的候选值
L_values = [9]  # 例如可以尝试的L值, 3, 5, 7, 9
S_values = [256]  # 例如可以尝试的S值8, 16, 32, 64, 128, 256

# 训练过程：遍历每一组L和S的组合
for L in L_values:
    for S in S_values:
        print(f"Training with L={L}, S={S}...")
        # 存储最佳模型和参数
        best_local_val_acc = 0.
        best_local_val_loss = 1.
        best_local_epoch = 0
        # 初始化模型
        net = SCN(L, S).to(device)  # 初始化SCN模型
        criterion = nn.MSELoss().to(device)  # 使用MSELoss
        optimizer = optim.Adam(net.parameters(), lr=0.001)  # 使用Adam优化器

        # 记录训练和验证的指标
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []
        metric = PECN_metrics()  # 评价指标

        net.train()  # 设置网络为训练模式
        # 训练过程
        # 训练过程
        for epoch in range(Epoch):
            # 训练
            running_loss = 0.
            running_acc = 0.
            net.train()  # 设置网络为训练模式
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU

                outputs = net(inputs)
                loss_value = criterion(outputs, labels.float())  # BCELoss要求标签是float类型
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                # 计算准确度
                running_loss += loss_value.item()
                running_acc += metric.multi_acc(outputs, labels)  # 使用修正后的multi_acc

            acc1 = running_acc / (i + 1)
            loss1 = running_loss / (i + 1)
            train_acc.append(acc1)
            train_loss.append(loss1)

            # 验证
            net.eval()  # 设置网络为评估模式
            running_loss = 0.
            running_acc = 0.
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input_val, label_val = data
                    input_val, label_val = input_val.to(device), label_val.to(device)  # 将数据移到GPU
                    val_output = net(input_val)
                    running_loss += criterion(val_output, label_val.float()).item()  # BCELoss要求标签是float类型
                    running_acc += metric.multi_acc(val_output, label_val)

                acc2 = running_acc / (i + 1)
                loss2 = running_loss / (i + 1)
                val_acc.append(acc2)
                val_loss.append(loss2)

            if acc2 >= best_local_val_acc:
                best_local_val_acc = acc2
                best_local_epoch  = epoch
                torch.save(net.state_dict(), model_save_path)

            # 打印当前epoch的训练和验证结果
            print(f'Epoch {epoch + 1}/{Epoch} | Train Loss: {loss1:.5f} | Train Acc: {acc1:.5f} | '
                  f'Val Loss: {loss2:.5f} | Val Acc: {acc2:.5f}')

            # list = [loss1, acc1, loss2, acc2]
            # data = pd.DataFrame([list])
            # data.to_csv('C:\\Users\\HD\\Desktop\\nlnet.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
        # 每次训练结束后打印当前超参数组合的结果
        # print(f"Finished training for L={L}, S={S} with Best Local Val Loss: {best_local_val_loss:.5f} and Best Local Val Acc: {best_local_val_acc:.5f}")


# # 最终保存表现最好的模型
# if best_model is not None:
#     torch.save(best_model, model_save_path)
#     print(f"Best model saved for L={best_L}, S={best_S} with Best Val Loss: {best_val_loss:.5f} and Best Val Acc: {best_val_acc:.5f}")

