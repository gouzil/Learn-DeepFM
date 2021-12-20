import os
import time
from visualdl import LogReader
import matplotlib.pyplot as plt

# reader = None

# 开启visualDL_log才能输出可视化图像 [!请修改]
file_dir = "/home/aistudio/work/PaddleRec/models/rank/deepfm/visualDL_log/"

now_time = time.strftime('%Y-%m-%d',time.localtime(time.time()))

os.makedirs(f"./Visualize/img/{now_time}/")

# 获取最新文件
def new_report(path_):
    lists = os.listdir(path_)   # 列出目录的下所有文件和文件夹保存到lists
    lists.sort(key=lambda fn: os.path.getmtime(path_ + "/" + fn))  # 按时间排序
    file_new = os.path.join(path_, lists[-1])     # 获取最新的文件保存到file_new
    # print(file_new)
    return file_new

# 读取文件
try:
    train_path = new_report(file_dir + "train")
    # 根据实际路径修改log文件位置
    reader = LogReader(file_path=train_path)
    # 显示标签
    tags = reader.get_tags()
    print(tags)
    # 读取数据
    data1 = reader.get_data('scalar', 'train/auc')
    data2 = reader.get_data('scalar', 'train/loss')
except:
    print("读取train异常")
    pass

try:
    infer_path = new_report(file_dir + "infer")
    # 预测日志
    reader_1 = LogReader(file_path=infer_path)
    tags_1 = reader_1.get_tags()
    print(tags_1)
    data3 = reader_1.get_data('scalar', 'infer/auc')
except:
    print("读取infer异常")
    pass


auc_list = []
id_1 = []
for i in data1:
    auc_list.append(i.value)
    id_1.append(i.id)

plt.title("auc", fontsize = 24)
plt.plot(id_1, auc_list, linewidth=2)
path_1 = f"./Visualize/img/{now_time}/train-auc.png"
plt.savefig(path_1)
plt.show()
sum(auc_list)
average=sum(auc_list)/len(auc_list)
print('auc_list累加值',sum(auc_list),'平均值',average)
print(auc_list[-1])

loss_list = []
id_2 = []
for i in data2:
    loss_list.append(i.value)
    id_2.append(i.id)

plt.title("loss", fontsize = 24)
plt.plot(id_2, loss_list, linewidth=2)
path_2 = f"./Visualize/img/{now_time}/train-loss.png"
plt.savefig(path_2)
plt.show()
sum(loss_list)
average=sum(loss_list)/len(loss_list)
print('loss_list累加值',sum(loss_list),'平均值',average)
print(loss_list[-1])

infer_auc = []
id_3 = []
for i in data3:
    infer_auc.append(i.value)
    id_3.append(i.id)

plt.title("infer_auc", fontsize = 24)
plt.plot(id_3, infer_auc, linewidth=2)
path_3 = f"./Visualize/img/{now_time}/infer-auc.png"
plt.savefig(path_3)
plt.show()
sum(infer_auc)
average=sum(infer_auc)/len(infer_auc)
print('infer_auc累加值',sum(infer_auc),'平均值',average)
print(infer_auc[-1])
