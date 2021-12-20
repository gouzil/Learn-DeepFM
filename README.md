# Learn-DeepFM
本项目使用推荐系统中的多个模型进行对比，并基于Criteo数据集完成对点击率模型的训练和预测。也是对点击率模型的学习。

## 内容
* [项目说明](#项目说明)
* [安装说明](#安装说明)
* [数据准备](#数据准备)
* [模型选择](#模型选择)
* [模型训练](#模型训练)
* [模型评估](#模型评估)
* [模型预测](#模型预测)
* [模型导出](#模型导出)
* [模型推理](#模型推理)
* [模型优化](#模型优化)
* [模型部署](#模型部署)
* [文件清理](#文件清理)
* [已知问题](#已知问题)
* [待完善](#待完善)

<a name="项目说明"></a>

## 1 项目说明

据统计，2019年全国广告市场总体规模达8674.28亿元，较上年增长了8.54%（见图1），占国民生产总值（GDP）的0.88%。如今广告成为各短视频平台最重要的收入来源。

<center><img src='./doc/imgs/StatisticalData.png' width=600></center><center>图1.2009—2019中国广告经营额</center> 


针对短视频、搜索、资讯等场景，应用飞桨[PaddleRec](https://github.com/PaddlePaddle/PaddleRec/tree/master)的推荐算法技术，对召回数据进行排序最终展现给用户，最大限度吸引用户、留存用户、增加用户粘性、提高用户转化率。
<center><img src='./doc/imgs/logo.png' width=600></center>
<center><img src='./doc/imgs/structure.png' width=600></center>
<center><img src='./doc/imgs/overview.png' width=600></center>

<br/>
<br/>

#### **推荐系统大致流程**


<center><img src='./doc/imgs/circuit.png' width=600></center>

#### **方案难点**

* **推理速度要求高：** 在搜索中进行快速响应，增强用户使用体验，对模型推理速度有较高要求。
* **推理准确度要求：** 在实际应用中推荐内容是否准确，对模型的精确度有一定的要求。
* **推理召回率：** 根据用户的行为进行分析生成候选，再进行排序，最后呈现给用户。 
* **在离线一致性：** 如下图

<center><img src='./doc/imgs/whole_process.png' width=600></center>


<a name="安装说明"></a>


## 2安装说明

#### 环境要求

* PaddlePaddle >=2.0
* Python >= 3.7
* 操作系统: Windows/Mac/Linux

  > Windows下PaddleRec目前仅支持单机训练，分布式训练建议使用Linux环境
  
### 安装Paddle

- gpu环境pip安装
  ```bash
  python -m pip install paddlepaddle-gpu==2.0.0 
  ```
- cpu环境pip安装
  ```bash
  python -m pip install paddlepaddle # gcc8 
  ```
更多版本下载可参考paddle官网[下载安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/index_cn.html)

### 下载PaddleRec

注意：官方维护github版本地址：  
[https://github.com/PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec)

```bash
git clone https://github.com/PaddlePaddle/PaddleRec/
cd PaddleRec
```

<br/>

#### 大致的文件分布

```bash
aistudio@jupyter-885527-3178715:~$ tree -L 3
.
├── 3178715.ipynb
├── AutoLog
│   ├── auto_log
│   │   ├── autolog.h
│   │   ├── autolog.py
│   │   ├── device.py
│   │   ├── env.py
│   │   ├── __init__.py
│   │   ├── lite_autolog.h
│   │   ├── __pycache__
│   │   └── utils.py
│   ├── auto_log.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── build
│   │   ├── bdist.linux-x86_64
│   │   └── lib
│   ├── CMakeLists.txt
│   ├── dist
│   │   └── auto_log-1.0.0-py3-none-any.whl
│   ├── output
│   ├── README.md
│   ├── requirements.txt
│   └── setup.py
├── output_model_all_deepfm
│   └── 0
│       ├── rec.pdopt
│       ├── rec.pdparams
│       ├── tostatic.pdiparams
│       ├── tostatic.pdiparams.info
│       └── tostatic.pdmodel
└── PaddleRec
    ├── datasets
    ├── doc
    ├── __init__.py
    ├── LICENSE
    ├── models
    ├── __pycache__
    ├── README_EN.md
    ├── README.md
    ├── recserving
    ├── tests
    └── tools
```


<a name="数据准备"></a>


## 3 数据准备

本案例使用数据集使用官方demo提供的数据集```./PaddleRec/models/rank/deepfefm/data/sample_data/train/sample_train.txt```，格式如下：

```
click:0 dense_feature:0.0 dense_feature:0.00497512437811 ... dense_feature:0.08 1:737395 2:210498 ... 26:306163
```

其中```click```表示是否被点击，点击用1表示，未点击用0表示。```dense_feature```代表连续特征值，共13个。```1```代表离散特征，共26个。相邻特征使用```\t```分隔，缺失用空格表示。


引用官方readme，示例文件为```PaddleRec/datasets/criteo/slot_train_data_full/part-0```：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  

```bash
# criteo全量数据集下载(4.2G)
cd PaddleRec/datasets/criteo
sh run.sh
cd ~/
```

<a name="模型选择"></a>

## 4 模型选择


|       数据集        |       模型       |       train_loss        |       train_auc          |       acc         |
| :------------------: | :--------------------: | :---------: |:---------: | :---------: |
|       Criteo        |       DNN       |       0.44        |        0.79         |       0.79          |
|       Criteo        |       Logistic Regression       |       --        |      0.67          |       --          |
|       Criteo        |       FM       |       --        |       0.78          |       --          |
|       Criteo        |       GateDnn       |       --        |       0.79          |       --          |
|       Criteo        |       DeepFM       |       0.44797        |       0.78          |       --          |
|       criteo        |       Wide&Deep       |       0.76195         |       0.82          |       --          |
|       criteo        |       dcn       |       --         |       0.77          |       --          |
|       criteo        |       deepfefm       |       --         |       0.8028          |       --          |
|       criteo        |       DLRM       |       --         |       0.79          |       --          |
|       criteo        |       ffm       |       --         |       0.79          |       --          |
|       criteo        |       xDeepFM       |       --         |       0.79          |       --          |


<a name="模型训练"></a>


## 5 模型训练


本项目采用DeepFM作为点击率的模型，模型训练需要经过如下环节：

<center><img src='./doc/imgs/FlowChart.png' width=600></center>

[自定义数据集及Reader](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/custom_reader.md)、[自定义模型](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/model_develop.md)、[yaml文件配置](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/yaml.md)

具体代码请参考```PaddleRec/models/rank/deepfm/config.yaml```，可修改参数：

**runner变量**

|             名称              |     类型     |             取值                 | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :--------------------------------: | :------: | :------------------------------------------------------------------: |
|         train_data_dir          |    string    |                       任意                        |    是    |                        指定训练数据目录                        |
|         train_reader_path          |    string    |                       任意                        |    是    |            指定训练时用的Reader()所在python文件地址            |
|         train_batch_size            |    int    |                       >= 1                       |    是    |                   指定train阶段的批训练样本数量                    |
|         model_save_path            |    string    |                       任意                       |    是    |                 指定train阶段完成后Save参数的地址                  |
|         test_data_dir              |    string    |                       任意                        |    是    |                        指定测试数据目录                        |
|         infer_reader_path          |    string    |                       任意                        |    是    |                指定测试时用的Reader()所在python文件地址                |
|         infer_batch_size            |    int    |                       >= 1                      |    是    |                   指定infer阶段的批训练样本数量                    |
|         infer_load_path            |    string    |                       任意                       |    是    |                 指定infer阶段开始时初始化模型地址                 |
|         infer_start_epoch            |    int    |                       >= 0                       |    是    |    初始化模型时从第几个epoch保留的参数开始加载（从0开始计数，包括本次）    |
|         infer_end_epoch            |    int    |                           >= 0                           |    是    |    初始化模型时到第几个epoch保留的参数停止加载（从0开始技术，不包括本次）    |
|         use_gpu            |    bool    |                  True/False                   |    是    |               指定是否使用gpu，若为False则默认使用cpu                |
|         epochs            |    int    |                       >= 1                       |    是    |                   指定train阶段需要训练几个epoch                    |
|         print_interval            |    int    |                       >= 1                       |    是    |                   训练指标打印batch间隔                    |
|         use_auc            |    bool    |                       True/False                       |    否    |                   在每个epoch开始时重置auc指标的值                    |
|         use_visual            |    bool    |                     True/False                      |    否    |                开启模型训练的可视化功能，开启时需要安装visualDL                   |
|         use_inference            |    bool    |                       True/False                       |    否    |                 是否使用save_inference_model接口保存                  |
|         save_inference_feed_varnames         |    list[string]    |                组网中指定Variable的name                 |    否    |                 预测模型的入口变量name                 |
|         save_inference_fetch_varnames         |    list[string]    |                组网中指定Variable的name                  |    否    |                 预测模型的出口变量name               |
|         use_fleet         |    bool    |                  True/False                  |    否    |                 指定是否使用分布式运行单机多卡或多机多卡                 |
|         reader_type         |    string    |                  QueueDataset/DataLoader/CustomizeDataLoader                |    否    |                 指定使用的reader类型                 |
|         model_init_path         |    string    |                  任意                  |    否    |                 指定是否使用热启动，在训练初期加载初始化模型                 |


**hyper_parameters变量**
|          名称           |  类型  |       取值       | 是否必须 |          作用描述           |
| :---------------------: | :----: | :--------------: | :------: | :-------------------------: |
|     optimizer.class     | string | SGD/Adam/Adagrad |    是    |       指定优化器类型        |
| optimizer.learning_rate | float  |       > 0        |    否    |         指定学习率          |
|           reg           | float  |       > 0        |    否    | L2正则化参数，只在SGD下生效 |
|         others          |   /    |        /         |    /     |   由各个模型组网独立指定    |


【名词解释】

* 动态图：在这种模式下，每次执行一个运算，可以立即得到结果（而不是事先定义好网络结构，然后再执行），PaddlePaddle2.0开始默认使用动态图模式[paddle动态图模型预测](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/quick_start/dynamic_graph.html#dongtaitu)。
* 静态图：静态图需要先构建再运行，优势是在运行前可以对图结构进行优化，比如常数折叠、算子融合等，可以获得更快的前向运算速度。[Paddle静态图预测部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/export_model/basic_concept_cn.html#sanjingtaituyucebushu)


<br/>

#### 修改config

添加```use_visual: True```，用于模型可视化

官方demo示例：

```
runner:
  train_data_dir: "data/sample_data/train"
  train_reader_path: "criteo_reader" # importlib format
  use_gpu: True
  use_auc: True
  use_visual: True
  train_batch_size: 2
  epochs: 3
  print_interval: 2
...
```

全量数据，```config_bigdata.yaml```示例：

```
runner:
  train_data_dir: "../../../datasets/criteo/slot_train_data_full"
  train_reader_path: "criteo_reader" # importlib format
  use_gpu: True
  use_auc: True
  use_visual: True
  train_batch_size: 512
  epochs: 1
  print_interval: 10
...
```
 * 本文以动态图作为教程静态图等待补充 
```bash
# 先跑个官方demo
# 动态图训练
python -u ~/PaddleRec/tools/trainer.py -m ~/PaddleRec/models/rank/deepfm/config.yaml 
```
```bash
# 静态图训练
python -u ~/PaddleRec/tools/static_trainer.py -m ~/PaddleRec/models/rank/deepfm/config.yaml
```

```bash
# 再跑个criteo全量数据集 (此数据集可能要跑2小时!!!)
# 动态图训练
python -u ~/PaddleRec/tools/trainer.py -m ~/PaddleRec/models/rank/deepfm/config_bigdata.yaml
```

```output_model_all_deepfm_epochs_4/```为训练四次的结果


<a name="模型预测"></a>


## 6 模型预测

大致流程:

<img src='./doc/imgs/Model prediction process.png' width=500>

 * 官方demo小量数据集

```bash
# 动态图预测
python -u ~/PaddleRec/tools/infer.py -m ~/PaddleRec/models/rank/deepfm/config.yaml
```

```bash
# 静态图预测
python -u ~/PaddleRec/tools/static_infer.py -m ~/PaddleRec/models/rank/deepfm/config.yaml
```

 * criteo全量数据集

```bash
# 动态图预测
python -u ~/PaddleRec/tools/infer.py -m ~/PaddleRec/models/rank/deepfm/config_bigdata.yaml
```

```bash
# 静态图预测
python -u ~/PaddleRec/tools/static_infer.py -m ~/PaddleRec/models/rank/deepfm/config_bigdata.yaml
```


<a name="模型评估"></a>


## 7 模型评估

#### AUC

AUC面积（Area Under Curve），又称ROC曲线下的面积，它描述的是分类器C随机抽取的一个正例的预测概率大于一个负例的预测概率的概率。简单地说，就是做随机抽样时，P(P) ≥ P(N)中 ≥ 成立的概率。

<img src='https://ai-studio-static-online.cdn.bcebos.com/1b19c943434a45c88fa6aafa123a39849160489cbed8403c80d9761e1eb4a2bb' width=400>

如上图紫色部分，AUC面积是指ROC曲线与下方坐标轴围成图形的面积，因为ROC曲线一般在斜线的上方，所以AUC面积的取值为[0.5，1]，数值越大表示模型分类效果越好。同时，AUC面积和ROC曲线一样，不受正负样例分布的影响。其常用评估界值如下：

0.5 - 0.7：效果较低

0.7 - 0.85：效果一般

0.85 - 0.95：效果很好

0.95 - 1：效果非常好，但很可能过拟合，一般不太可能



| 模型 | auc | 预测时间 | 总结 |
| :------| :------| :------ | :------ | 
| deepFM | 0.78 | 279.38s |  效果一般 | 


<a name="模型可视化"></a>


## 8 模型可视化

根据日志目录的不同修改```[!请修改]```进行代码的修改```code/Visualize.py```文件

```bash
python code/Visualize.py
```

![](./doc/imgs/deepfm_infer_auc.png)
![](./doc/imgs/deepfm_train_auc.png)
![](./doc/imgs/deepfm_train_loss.png)


<a name="模型推理"></a>


## 9 模型推理

本项目采用DeepFM作为点击率的模型，模型推理需要经过如下环节：

<img src='./doc/imgs/Reasoning process.png' width=600>

<br/>
<br/>

【可能会用上的文档】

[1] [Linux端基础训练预测功能测试](https://github.com/PaddlePaddle/PaddleRec/blob/master/test_tipc/doc/test_train_inference_python.md)

```bash
# 保存模型
python -u ~/PaddleRec/tools/to_static.py -m ~/PaddleRec/models/rank/deepfm/config_bigdata_init.yaml
```

```bash
# 需要安装的库：
pip install pynvml psutil GPUtil

# 由于GitHub及其难下载，所以在aistudio内置此模块
# git clone https://github.com/LDOUBLEV/AutoLog.git
%cd ~/AutoLog
pip install -r requirements.txt
python setup.py bdist_wheel
pip install ./dist/auto_log-1.0.0-py3-none-any.whl
cd ../
```

 * 推理(criteo数据集)

```bash
python -u ~/PaddleRec/tools/paddle_infer.py --model_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdmodel --params_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdiparams --use_gpu=True --data_dir=./PaddleRec/models/rank/deepfm/data/sample_data/train --reader_file=./PaddleRec/models/rank/deepfm/criteo_reader.py --batchsize=5
```

**不同模式速度测试**

测试环境：

OS_version: Ubuntu 16.04

CUDA_version: 10.1.243

CUDNN_version: 7.3.1

drivier_version: 418.67

CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 4核

GPU: NVIDIA Tesla V100 SXM2 32GB

| 是否启用GPU | 是否开启tensorRT加速 | 是否开启mkldnn加速 | 线程数 | 批大小 | 是否开启内存优化器 | 开启IR优化 |         时长        | 
| --------- | ------------------ | ---------------- | ----- | ----- | --------------- | --------- | ------------------- |
| False     |        False       |        False     | 1     | 5     |       True      |   True    |   14分钟50秒482毫秒  |
| True     |        False       |        False     | 1     | 5     |       True      |   True    |   16分钟5秒771毫秒  |
| False     |        False       |        False     | 128     | 5     |       True      |   True    | 1小时12分钟15秒739毫秒 |
| False     |        False       |        True     | 1     | 5     |       True      |   True    |  18分钟2秒37毫秒 |
| False     |        False       |        False     | 1     | 2048  |       True      |   True    | 4分钟32秒14毫秒  |
| True     |        False       |        False     | 1     | 2048  |       True      |   True    | 4分钟14秒394毫秒  |

组合太多了，不一一跑了，欢迎补充。

详细的日志存储在```doc/ReasoningTest/results.md```文件中


<a name="模型优化"></a>


## 10 模型优化


在本项目示例中，仅使用默认配置进行了一轮的训练，模型还称不上训练到最优，已经有了基本的效果，如何继续优化呢？

<br/>

#### 调整超参

在PaddleRec中，我们已经将超参数都写在config.yaml中，所以只需要对config.yaml一个文件进行修改，就能够清晰的对比模型效果，并快速进行模型效果验证，极大的提升模型的迭代效率。

**增加训练轮数**

在训练模型的时候，效果较差可能是因为欠拟合引起的。我们可以增加训练的轮数，让模型获得更充分的训练，以此来提高模型的效果。

随着epoch 数量的增加， 权重更新迭代的次数增多， 曲线从最开始的不拟合状态， 进入优化拟合状态， 最终进入过拟合。



以本教程中```~/PaddleRec/models/rank/deepfm/config_bigdata.yaml```为例：

<img src='./doc/imgs/Increase_training.png' width=400>


| 迭代次数 | train_loss | train_auc | train_auc |
| ------ | ----------- | ------- | ------- |
|  1     |   0.44797   |   0.78  |  0.77214  |
|  4     |   0.47313   |   0.81  |  0.80526  |
|  10    |   0.46200   |   0.81  |  0.76423  |

**修改批大小**

在训练模型的时候，我们可以通过修改批大小，让模型获得更充分的训练，以此来提高模型的效果。
直观的理解：
Batch Size定义：一次训练所选取的样本数。
Batch Size的大小影响模型的优化程度和速度。同时其直接影响到GPU显存或者内存的使用情况，假如你GPU显存或者内存不大，该数值最好设置小一点。

以本教程中```~/PaddleRec/models/rank/deepfm/config_bigdata.yaml```为例：

<img src='./doc/imgs/Revise_batch_size.png' width=400>

| 批大小 | train_loss | train_auc | train_auc |
| ------ | ----------- | ------- | ------- |
|  256   |   0.48464   |   0.77  |  0.75091  |
|  512   |   0.44797   |   0.78  |  0.77214  |
|  1024  |   0.48464   |   0.77  |  0.77433  |

**更换优化器**

在训练模型的时候，我们可以更换优化器，尝试不同的学习率以求获得提升。在PaddleRec中，我们提供SGD/Adam/AdaGrad优化器供您尝试。也可以通过learning_rate选项修改学习率。

* Adagrad它利用迭代次数和累积梯度，对学习率进行自动衰减，2011年提出。从而使得刚开始迭代时，学习率较大，可以快速收敛。而后来则逐渐减小，精调参数，使得模型可以稳定找到最优点。

* SGD全称Stochastic Gradient Descent，随机梯度下降，1847年提出。每次选择一个mini-batch，而不是全部样本，使用梯度下降来更新模型参数。它解决了随机小批量样本的问题，但仍然有自适应学习率、容易卡在梯度较小点等问题。

* Adam是SGDM和RMSProp的结合，它基本解决了之前提到的梯度下降的一系列问题，比如随机小样本、自适应学习率、容易卡在梯度较小点等问题。

仍然以本教程中```~/PaddleRec/models/rank/deepfm/config_bigdata.yaml```为例：

<img src='./doc/imgs/optimizer.png' width=400>

|   优化器  |  train_loss | train_auc | infer_auc |
| -------- | ----------- | -------- | --------- |
|  Adam    |   0.44797   |   0.78   |  0.77214 |
|  SGD     |   0.47405   |   0.77   |  0.76693 |
|  AdaGrad |   0.47380   |   0.77   |  0.76598 |

**修改学习率**

也可以通过learning_rate选项修改学习率。

 * 学习率(Learning rate)作为监督学习以及深度学习中重要的超参，其决定着目标函数能否收敛到局部最小值以及何时收敛到最小值。合适的学习率能够使目标函数在合适的时间内收敛到局部最小值。

仍然以本教程中```~/PaddleRec/models/rank/deepfm/config_bigdata.yaml```为例：

<img src='./doc/imgs/optimizer.png' width=400>

| 学习率 | train_loss | train_auc | infer_auc |
| ------ | ----------- | ------- | --------- |
|  0.001    |   0.44797   |   0.78  |  0.77214 |
|  0.01     |   2.96769   |   0.51  |  0.49999 |
|  0.0001   |   0.46342   |   0.77  |  0.78671 |


**调整全连接层**

在训练模型的时候，我们可以很方便的指定模型的全连接层共有几层，以及每一层的维度。

 * 全连接层，是每一个结点都与上一层的所有结点相连，用来把前边提取到的特征综合起来。由于其全相连的特性，一般全连接层的参数也是最多的。

仍然以本教程中```~/PaddleRec/models/rank/deepfm/config_bigdata.yaml```为例：

<img src='https://ai-studio-static-online.cdn.bcebos.com/8106c548a6f94a41b8af006870e33663a0ae0e92f36b42b7bf0cbbf77adfcc87' width=600>

|        全连接层      |  train_loss | train_auc | infer_auc |
| ------------------- | ----------- | ------- | --------- |
|  [400, 400, 400]    |   0.44797   |   0.78  |  0.77214 |
|  [512, 256, 128]    |   0.47318   |   0.77  |  0.76565 |
|  [1024, 1024, 1024] |   0.47412   |   0.77  |  0.76702 |

<a name="模型部署"></a>


## 11 模型部署

**大致流程:**

<img src='./doc/imgs/deepfm_server.png' width=800>

<br/>

 * post参数:

<img src='./doc/imgs/post_deepfm_0.png' width=800>

<br/>
<br/>

 * post Header参数:

<img src='./doc/imgs/post_deepfm_1.png' width=800>

 * post 文件参数(键记得改为文件类型):

<img src='./doc/imgs/post_deepfm_2.png' width=800>

#### 启动服务

在终端执行:
```bash
cd Docker/
sh start.sh
```

```python
# 执行post
import requests
 
url = 'http://127.0.0.1:8867/upload'
files = {'file': open('./PaddleRec/models/rank/deepfm/data/sample_data/train/sample_train.txt', 'rb')}           
params = {"debug":"true", "user_id":"11", "full":"false","re_model":"json","save":"false"}
  
response = requests.post(url, params=params, files=files)
json = response.text
print(json)
```

<br/>

返回值示例：

~~~json
{"code": 200, "dtype": "float32", "data": [[[0.1726774275302887], [0.6314680576324463], [0.030182218179106712], [0.04229707643389702], [0.06759048998355865]], [[0.08457722514867783], [0.6918926239013672], [0.19035762548446655], [0.4685360789299011], [0.38435056805610657]], [[0.7631178498268127], [0.35949385166168213], [0.07950007915496826], [0.025452962145209312], [0.017600852996110916]], [[0.10469464957714081], [0.069789819419384], [0.08838725090026855], [0.29404881596565247], [0.7731786370277405]], [[0.17364516854286194], [0.3019689619541168], [0.23101796209812164], [0.19991131126880646], [0.060729727149009705]], [[0.4240683615207672], [0.0025945627130568027], [0.3556656539440155], [0.02395329251885414], [0.012945180758833885]], [[0.06743166595697403], [0.5076236724853516], [0.2516230642795563], [0.30182886123657227], [0.16652143001556396]], [[0.2995298206806183], [0.2332691252231598], [0.06336560845375061], [0.3804313540458679], [0.017034510150551796]], [[0.14554713666439056], [0.04525890201330185], [0.015055081807076931], [0.02107999473810196], [0.12089382857084274]], [[0.4439798593521118], [0.31729280948638916], [0.2385992705821991], [0.09055238962173462], [0.08244699984788895]], [[0.05468741059303284], [0.1615753024816513], [0.6356972455978394], [0.11687539517879486], [0.5943728089332581]], [[0.8153748512268066], [0.6498715281486511], [0.17078666388988495], [0.23964233696460724], [0.18998616933822632]], [[0.5524210333824158], [0.4712432622909546], [0.45373600721359253], [0.37936902046203613], [0.5279058218002319]], [[0.24596500396728516], [0.46170666813850403], [0.1550101637840271], [0.3604608178138733], [0.5804333090782166]], [[0.1904362291097641], [0.10485503822565079], [0.22313614189624786], [0.49750977754592896], [0.6349889039993286]], [[0.13100089132785797], [0.09884602576494217], [0.009975193999707699], [0.09688897430896759], [0.2861407995223999]]], "user_id": 77}
~~~

<a name="文件清理"></a>

## 12 文件清理

```bash
# 清理数据集
cd ~/
rm -rf ~/PaddleRec/datasets/criteo/slot_train_data_full.tar.gz
rm -rf ~/PaddleRec/datasets/criteo/slot_test_data_full.tar.gz
rm -rf ~/PaddleRec/datasets/criteo/slot_train_data_full
rm -rf ~/PaddleRec/datasets/criteo/slot_test_data_full
```

<a name="已知问题"></a>

## 13 已知问题

 * [1] 改进后的模型部署，日志会重复输出(暂时用强制退出解决)

 * [2] 在aistuido平台使用模型部署时使用```--enable_tensorRT```进行加速时提示```请使用带有TensorR编译的Paddle推断库```[使用Paddle-TensorRT库预测](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/advanced_guide/performance_improving/inference_improving/paddle_tensorrt_infer.html)

<a name="待完善"></a>

## 14 待完善

 * [1] 模型优化数据

 * [2] 模型部署优化速度

 * [3] docker服务编写

<a name="参考资料"></a>

## 参考资料

**排名不分先后:**

 * [1] 2019中国广告年度数据报告: [http://mlzg.shiyan.gov.cn/html/2020/shiyan_whjy_0325/17402.html](http://mlzg.shiyan.gov.cn/html/2020/shiyan_whjy_0325/17402.html)

 * [2] PaddleRec: [https://github.com/PaddlePaddle/PaddleRec/tree/master](https://github.com/PaddlePaddle/PaddleRec/tree/master)

 * [3]基于 DeepFM 模型的点击率预估模型: [https://aistudio.baidu.com/aistudio/projectdetail/2251589?channelType=0&channel=0](https://aistudio.baidu.com/aistudio/projectdetail/2251589?channelType=0&channel=0)

 * [4]私人电影推荐: [https://aistudio.baidu.com/aistudio/projectdetail/1481839?channelType=0&channel=0](https://aistudio.baidu.com/aistudio/projectdetail/1481839?channelType=0&channel=0)

 * [5]神经网络中Batch Size的理解: [https://blog.csdn.net/qq_34886403/article/details/82558399](https://blog.csdn.net/qq_34886403/article/details/82558399)

 * [6]机器学习2 -- 优化器（SGD、SGDM、Adagrad、RMSProp、Adam）: [https://zhuanlan.zhihu.com/p/208178763](https://zhuanlan.zhihu.com/p/208178763)

 * [7]全连接层: [https://baike.baidu.com/item/%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82](https://baike.baidu.com/item/%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82)

<a name="开源数据"></a>

## 开源数据

 * 非常感谢[PaddlePaddle](https://github.com/PaddlePaddle)和[Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge/)开源的数据集