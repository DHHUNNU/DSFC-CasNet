# -----#
# author:HD
# year&month&day:2025:02:24
# -----#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from utils.SNSNet_dataread_train import SNSNet_dataread
from utils.SNSN_metrics import SNSN_metrics
from network.SNSN import SNSN
import pandas as pd
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
model_save_path = 'model/SNSNet.pth'

# 数据路径
file_path = 'datasets/train_datasets/'
data_name_train = 'EL22208A'
data_name_val = 'EL22208A'
num = 4096
# 读取训练数据
train_data = SNSNet_dataread(file_path, data_name_train, label_num=num)
a, b, c = train_data.data_read(file_path, data_name_train,  label_num=num)

train_loader = DataLoader(dataset=train_data,
                          batch_size=Batch,
                          shuffle=True,
                          pin_memory=False,
                          drop_last=True)
# 读取验证数据
val_data = SNSNet_dataread(file_path, data_name_val, label_num=num)  # 添加segment_type和segment_length
val_loader = DataLoader(dataset=val_data,
                        batch_size=1024,
                        shuffle=True,
                        pin_memory=False,
                        drop_last=True)


# 网格搜索参数：L和S的候选值
L_values = [9]  # 例如可以尝试的L值, 3, 5, 7, 9
S_values = [128]  # 例如可以尝试的S值8, 16, 32, 64, 128, 256

# 训练过程：遍历每一组L和S的组合
for L in L_values:
    for S in S_values:
        print(f"Training with L={L}, S={S}...")
        # 存储最佳模型和参数
        best_local_val_ncc = 0.
        best_local_val_loss = float('inf')
        best_local_epoch = 0
        # 初始化模型
        net = SNSN(L, S).to(device)  # 初始化SCN模型
        criterion = nn.MSELoss().to(device)  # 使用MSELoss
        optimizer = optim.Adam(net.parameters(), lr=0.001)  # 使用Adam优化器

        # 记录训练和验证的指标
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []
        metric = SNSN_metrics()  # 评价指标

        net.train()  # 设置网络为训练模式
        # 训练过程
        for epoch in range(Epoch):
            # 训练
            running_loss = 0.
            running_ncc = 0.
            net.train()  # 设置网络为训练模式
            for i, data in enumerate(train_loader):
                inputs, labels, profiles = data
                inputs, labels, profiles = inputs.to(device), labels.to(device), profiles.to(device)  # 将数据移到GPU
                outputs = net(inputs, labels)
                # print(outputs.size())
                loss_value = criterion(outputs, profiles.float())  # BCELoss要求标签是float类型
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                # 计算准确度
                running_loss += loss_value.item()
                running_ncc += metric.NCC(outputs, profiles)  # 使用修正后的multi_acc

            ncc1 = running_ncc / (i + 1)
            loss1 = running_loss / (i + 1)
            train_acc.append(ncc1)
            train_loss.append(loss1)

            # 验证
            net.eval()  # 设置网络为评估模式
            running_loss = 0.
            running_ncc = 0.
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input_val, label_val, profile_val = data
                    input_val, label_val, profile_val = input_val.to(device), label_val.to(device), profile_val.to(device)  # 将数据移到GPU
                    val_output = net(input_val, label_val)
                    running_loss += criterion(val_output, profile_val.float()).item()  # BCELoss要求标签是float类型
                    running_ncc += metric.NCC(val_output, profile_val)

                ncc2 = running_ncc / (i + 1)
                loss2 = running_loss / (i + 1)
                val_acc.append(ncc2)
                val_loss.append(loss2)

            if loss2 < best_local_val_loss:
                best_local_val_loss = loss2
                best_local_epoch = epoch
                torch.save(net.state_dict(), model_save_path)

            # 打印当前epoch的训练和验证结果
            print(f'Epoch {epoch + 1}/{Epoch} | Train Loss: {loss1:.5f} | Train Ncc: {ncc1:.5f} | '
                  f'Val Loss: {loss2:.5f} | Val Ncc: {ncc2:.5f}')
            #
            # list = [loss1, ncc1, loss2, ncc2]
            # data = pd.DataFrame([list])
            # data.to_csv('C:\\Users\\HD\\Desktop\\snsnet.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
# # 最终保存表现最好的模型
# if best_model is not None:
#     torch.save(best_model, model_save_path)
#     print(f"Best model saved for L={best_L}, S={best_S} with Best Val Loss: {best_val_loss:.5f} and Best Val Acc: {best_val_acc:.5f}")
