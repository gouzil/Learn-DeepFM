#### 测试环境
OS_version: Ubuntu 16.04

CUDA_version: 10.1.243

CUDNN_version: 7.3.1

drivier_version: 418.67

CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 4核

GPU: NVIDIA Tesla V100 SXM2 32GB
___
| 是否启用GPU | 是否开启tensorRT加速 | 是否开启mkldnn加速 | 线程数 | 批大小 | 是否开启内存优化器 | 开启IR优化 |         时长        | 
| --------- | ------------------ | ---------------- | ----- | ----- | --------------- | --------- | ------------------- |
| False     |        False       |        False     | 1     | 5     |       True      |   True    |   14分钟50秒482毫秒  |

~~~bash
python -u ~/work/PaddleRec/tools/paddle_infer.py --model_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdmodel --params_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdiparams --use_gpu=False --data_dir=/home/aistudio/work/PaddleRec/datasets/criteo/slot_test_data_full --reader_file=/home/aistudio/work/PaddleRec/models/rank/deepfm/criteo_reader.py --batchsize=5 --cpu_threads=1
~~~
日志：
~~~bash
[2021/12/15 15:05:26] root INFO: 
[2021/12/15 15:05:26] root INFO: ---------------------- Env info ----------------------
[2021/12/15 15:05:26] root INFO:  OS_version: Ubuntu 16.04
[2021/12/15 15:05:26] root INFO:  CUDA_version: 10.1.243
[2021/12/15 15:05:26] root INFO:  CUDNN_version: 7.3.1
[2021/12/15 15:05:26] root INFO:  drivier_version: 418.67
[2021/12/15 15:05:26] root INFO: ---------------------- Paddle info ----------------------
[2021/12/15 15:05:26] root INFO:  paddle_version: 2.2.1
[2021/12/15 15:05:26] root INFO:  paddle_commit: a5cf2e305b744e3ebd2f2210341f88d349d4ec5e
[2021/12/15 15:05:26] root INFO:  log_api_version: 1.0
[2021/12/15 15:05:26] root INFO: ----------------------- Conf info -----------------------
[2021/12/15 15:05:26] root INFO:  runtime_device: cpu
[2021/12/15 15:05:26] root INFO:  ir_optim: True
[2021/12/15 15:05:26] root INFO:  enable_memory_optim: True
[2021/12/15 15:05:26] root INFO:  enable_tensorrt: False
[2021/12/15 15:05:26] root INFO:  enable_mkldnn: False
[2021/12/15 15:05:26] root INFO:  cpu_math_library_num_threads: 1
[2021/12/15 15:05:26] root INFO: ----------------------- Model info ----------------------
[2021/12/15 15:05:26] root INFO:  model_name: rec_model
[2021/12/15 15:05:26] root INFO:  precision: None
[2021/12/15 15:05:26] root INFO: ----------------------- Data info -----------------------
[2021/12/15 15:05:26] root INFO:  batch_size: 5
[2021/12/15 15:05:26] root INFO:  input_shape: dynamic
[2021/12/15 15:05:26] root INFO:  data_num: 368123
[2021/12/15 15:05:26] root INFO: ----------------------- Perf info -----------------------
[2021/12/15 15:05:26] root INFO:  cpu_rss(MB): 298.3203, gpu_rss(MB): 10.0, gpu_util: 0.0%
[2021/12/15 15:05:26] root INFO:  total time spent(s): 213.3379
[2021/12/15 15:05:26] root INFO:  preprocess_time(ms): 0.1783, inference_time(ms): 0.3902, postprocess_time(ms): 0.0111
~~~
___
| 是否启用GPU | 是否开启tensorRT加速 | 是否开启mkldnn加速 | 线程数 | 批大小 | 是否开启内存优化器 | 开启IR优化 |         时长        | 
| --------- | ------------------ | ---------------- | ----- | ----- | --------------- | --------- | ------------------- |
| True     |        False       |        False     | 1     | 5     |       True      |   True    |   16分钟5秒771毫秒  |

~~~bash
python -u ~/work/PaddleRec/tools/paddle_infer.py --model_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdmodel --params_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdiparams --use_gpu=False --data_dir=/home/aistudio/work/PaddleRec/datasets/criteo/slot_test_data_full --reader_file=/home/aistudio/work/PaddleRec/models/rank/deepfm/criteo_reader.py --batchsize=5
~~~
日志：
~~~bash
[2021/12/15 15:23:41] root INFO: 
[2021/12/15 15:23:41] root INFO: ---------------------- Env info ----------------------
[2021/12/15 15:23:41] root INFO:  OS_version: Ubuntu 16.04
[2021/12/15 15:23:41] root INFO:  CUDA_version: 10.1.243
[2021/12/15 15:23:41] root INFO:  CUDNN_version: 7.3.1
[2021/12/15 15:23:41] root INFO:  drivier_version: 418.67
[2021/12/15 15:23:41] root INFO: ---------------------- Paddle info ----------------------
[2021/12/15 15:23:41] root INFO:  paddle_version: 2.2.1
[2021/12/15 15:23:41] root INFO:  paddle_commit: a5cf2e305b744e3ebd2f2210341f88d349d4ec5e
[2021/12/15 15:23:41] root INFO:  log_api_version: 1.0
[2021/12/15 15:23:41] root INFO: ----------------------- Conf info -----------------------
[2021/12/15 15:23:41] root INFO:  runtime_device: gpu
[2021/12/15 15:23:41] root INFO:  ir_optim: True
[2021/12/15 15:23:41] root INFO:  enable_memory_optim: True
[2021/12/15 15:23:41] root INFO:  enable_tensorrt: False
[2021/12/15 15:23:41] root INFO:  enable_mkldnn: False
[2021/12/15 15:23:41] root INFO:  cpu_math_library_num_threads: 1
[2021/12/15 15:23:41] root INFO: ----------------------- Model info ----------------------
[2021/12/15 15:23:41] root INFO:  model_name: rec_model
[2021/12/15 15:23:41] root INFO:  precision: None
[2021/12/15 15:23:41] root INFO: ----------------------- Data info -----------------------
[2021/12/15 15:23:41] root INFO:  batch_size: 5
[2021/12/15 15:23:41] root INFO:  input_shape: dynamic
[2021/12/15 15:23:41] root INFO:  data_num: 368123
[2021/12/15 15:23:41] root INFO: ----------------------- Perf info -----------------------
[2021/12/15 15:23:41] root INFO:  cpu_rss(MB): 2577.7227, gpu_rss(MB): 795.0, gpu_util: 4.0%
[2021/12/15 15:23:41] root INFO:  total time spent(s): 298.272
[2021/12/15 15:23:41] root INFO:  preprocess_time(ms): 0.3471, inference_time(ms): 0.4531, postprocess_time(ms): 0.0101
~~~

___

可能是实际线程数小于设置线程数导致的
| 是否启用GPU | 是否开启tensorRT加速 | 是否开启mkldnn加速 | 线程数 | 批大小 | 是否开启内存优化器 | 开启IR优化 |         时长        | 
| --------- | ------------------ | ---------------- | ----- | ----- | --------------- | --------- | ------------------- |
| False     |        False       |        False     | 128     | 5     |       True      |   True    | 1小时12分钟15秒739毫秒 |

~~~bash
python -u ~/work/PaddleRec/tools/paddle_infer.py --model_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdmodel --params_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdiparams --use_gpu=False --data_dir=/home/aistudio/work/PaddleRec/datasets/criteo/slot_test_data_full --reader_file=/home/aistudio/work/PaddleRec/models/rank/deepfm/criteo_reader.py --batchsize=5 --cpu_threads=128
~~~
日志：
~~~bash
[2021/12/15 16:49:23] root INFO: 
[2021/12/15 16:49:23] root INFO: ---------------------- Env info ----------------------
[2021/12/15 16:49:23] root INFO:  OS_version: Ubuntu 16.04
[2021/12/15 16:49:23] root INFO:  CUDA_version: 10.1.243
[2021/12/15 16:49:23] root INFO:  CUDNN_version: 7.3.1
[2021/12/15 16:49:23] root INFO:  drivier_version: 418.67
[2021/12/15 16:49:23] root INFO: ---------------------- Paddle info ----------------------
[2021/12/15 16:49:23] root INFO:  paddle_version: 2.2.1
[2021/12/15 16:49:23] root INFO:  paddle_commit: a5cf2e305b744e3ebd2f2210341f88d349d4ec5e
[2021/12/15 16:49:23] root INFO:  log_api_version: 1.0
[2021/12/15 16:49:23] root INFO: ----------------------- Conf info -----------------------
[2021/12/15 16:49:23] root INFO:  runtime_device: cpu
[2021/12/15 16:49:23] root INFO:  ir_optim: True
[2021/12/15 16:49:23] root INFO:  enable_memory_optim: True
[2021/12/15 16:49:23] root INFO:  enable_tensorrt: False
[2021/12/15 16:49:23] root INFO:  enable_mkldnn: False
[2021/12/15 16:49:23] root INFO:  cpu_math_library_num_threads: 128
[2021/12/15 16:49:23] root INFO: ----------------------- Model info ----------------------
[2021/12/15 16:49:23] root INFO:  model_name: rec_model
[2021/12/15 16:49:23] root INFO:  precision: None
[2021/12/15 16:49:23] root INFO: ----------------------- Data info -----------------------
[2021/12/15 16:49:23] root INFO:  batch_size: 5
[2021/12/15 16:49:23] root INFO:  input_shape: dynamic
[2021/12/15 16:49:23] root INFO:  data_num: 368123
[2021/12/15 16:49:23] root INFO: ----------------------- Perf info -----------------------
[2021/12/15 16:49:23] root INFO:  cpu_rss(MB): 299.957, gpu_rss(MB): 10.0, gpu_util: 0.0%
[2021/12/15 16:49:23] root INFO:  total time spent(s): 3535.5181
[2021/12/15 16:49:23] root INFO:  preprocess_time(ms): 0.2167, inference_time(ms): 9.3735, postprocess_time(ms): 0.0139
~~~
___
| 是否启用GPU | 是否开启tensorRT加速 | 是否开启mkldnn加速 | 线程数 | 批大小 | 是否开启内存优化器 | 开启IR优化 |         时长        | 
| --------- | ------------------ | ---------------- | ----- | ----- | --------------- | --------- | ------------------- |
| False     |        False       |        True     | 1     | 5     |       True      |   True    |  18分钟2秒37毫秒 |

~~~bash
python -u ~/work/PaddleRec/tools/paddle_infer.py --model_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdmodel --params_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdiparams --use_gpu=False --enable_mkldnn=True --data_dir=/home/aistudio/work/PaddleRec/datasets/criteo/slot_test_data_full --reader_file=/home/aistudio/work/PaddleRec/models/rank/deepfm/criteo_reader.py --batchsize=5
~~~
日志：
~~~bash
[2021/12/15 17:13:12] root INFO: 
[2021/12/15 17:13:12] root INFO: ---------------------- Env info ----------------------
[2021/12/15 17:13:12] root INFO:  OS_version: Ubuntu 16.04
[2021/12/15 17:13:12] root INFO:  CUDA_version: 10.1.243
[2021/12/15 17:13:12] root INFO:  CUDNN_version: 7.3.1
[2021/12/15 17:13:12] root INFO:  drivier_version: 418.67
[2021/12/15 17:13:12] root INFO: ---------------------- Paddle info ----------------------
[2021/12/15 17:13:12] root INFO:  paddle_version: 2.2.1
[2021/12/15 17:13:12] root INFO:  paddle_commit: a5cf2e305b744e3ebd2f2210341f88d349d4ec5e
[2021/12/15 17:13:12] root INFO:  log_api_version: 1.0
[2021/12/15 17:13:12] root INFO: ----------------------- Conf info -----------------------
[2021/12/15 17:13:12] root INFO:  runtime_device: cpu
[2021/12/15 17:13:12] root INFO:  ir_optim: True
[2021/12/15 17:13:12] root INFO:  enable_memory_optim: True
[2021/12/15 17:13:12] root INFO:  enable_tensorrt: False
[2021/12/15 17:13:12] root INFO:  enable_mkldnn: True
[2021/12/15 17:13:12] root INFO:  cpu_math_library_num_threads: 1
[2021/12/15 17:13:12] root INFO: ----------------------- Model info ----------------------
[2021/12/15 17:13:12] root INFO:  model_name: rec_model
[2021/12/15 17:13:12] root INFO:  precision: None
[2021/12/15 17:13:12] root INFO: ----------------------- Data info -----------------------
[2021/12/15 17:13:12] root INFO:  batch_size: 5
[2021/12/15 17:13:12] root INFO:  input_shape: dynamic
[2021/12/15 17:13:12] root INFO:  data_num: 368123
[2021/12/15 17:13:12] root INFO: ----------------------- Perf info -----------------------
[2021/12/15 17:13:12] root INFO:  cpu_rss(MB): 301.8281, gpu_rss(MB): 10.0, gpu_util: 0.0%
[2021/12/15 17:13:12] root INFO:  total time spent(s): 403.6152
[2021/12/15 17:13:12] root INFO:  preprocess_time(ms): 0.1746, inference_time(ms): 0.9097, postprocess_time(ms): 0.0121
~~~
___
注：增大批大小, 加快计算速度的同时会导致准确度降低
| 是否启用GPU | 是否开启tensorRT加速 | 是否开启mkldnn加速 | 线程数 | 批大小 | 是否开启内存优化器 | 开启IR优化 |         时长        | 
| --------- | ------------------ | ---------------- | ----- | ----- | --------------- | --------- | ------------------- |
| False     |        False       |        False     | 1     | 2048  |       True      |   True    | 4分钟32秒14毫秒  |

~~~bash
!python -u ~/work/PaddleRec/tools/paddle_infer.py --model_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdmodel --params_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdiparams --use_gpu=False --data_dir=/home/aistudio/work/PaddleRec/datasets/criteo/slot_test_data_full --reader_file=/home/aistudio/work/PaddleRec/models/rank/deepfm/criteo_reader.py --batchsize=2048
~~~
日志：
~~~bash
[2021/12/15 17:28:34] root INFO: 
[2021/12/15 17:28:34] root INFO: ---------------------- Env info ----------------------
[2021/12/15 17:28:34] root INFO:  OS_version: Ubuntu 16.04
[2021/12/15 17:28:34] root INFO:  CUDA_version: 10.1.243
[2021/12/15 17:28:34] root INFO:  CUDNN_version: 7.3.1
[2021/12/15 17:28:34] root INFO:  drivier_version: 418.67
[2021/12/15 17:28:34] root INFO: ---------------------- Paddle info ----------------------
[2021/12/15 17:28:34] root INFO:  paddle_version: 2.2.1
[2021/12/15 17:28:34] root INFO:  paddle_commit: a5cf2e305b744e3ebd2f2210341f88d349d4ec5e
[2021/12/15 17:28:34] root INFO:  log_api_version: 1.0
[2021/12/15 17:28:34] root INFO: ----------------------- Conf info -----------------------
[2021/12/15 17:28:34] root INFO:  runtime_device: cpu
[2021/12/15 17:28:34] root INFO:  ir_optim: True
[2021/12/15 17:28:34] root INFO:  enable_memory_optim: True
[2021/12/15 17:28:34] root INFO:  enable_tensorrt: False
[2021/12/15 17:28:34] root INFO:  enable_mkldnn: False
[2021/12/15 17:28:34] root INFO:  cpu_math_library_num_threads: 1
[2021/12/15 17:28:34] root INFO: ----------------------- Model info ----------------------
[2021/12/15 17:28:34] root INFO:  model_name: rec_model
[2021/12/15 17:28:34] root INFO:  precision: None
[2021/12/15 17:28:34] root INFO: ----------------------- Data info -----------------------
[2021/12/15 17:28:34] root INFO:  batch_size: 2048
[2021/12/15 17:28:34] root INFO:  input_shape: dynamic
[2021/12/15 17:28:34] root INFO:  data_num: 898
[2021/12/15 17:28:34] root INFO: ----------------------- Perf info -----------------------
[2021/12/15 17:28:34] root INFO:  cpu_rss(MB): 329.5977, gpu_rss(MB): 10.0, gpu_util: 0.0%
[2021/12/15 17:28:34] root INFO:  total time spent(s): 27.1381
[2021/12/15 17:28:34] root INFO:  preprocess_time(ms): 0.5514, inference_time(ms): 29.6432, postprocess_time(ms): 0.026
~~~
___

注：增大批大小, 加快计算速度的同时会导致准确度降低
| 是否启用GPU | 是否开启tensorRT加速 | 是否开启mkldnn加速 | 线程数 | 批大小 | 是否开启内存优化器 | 开启IR优化 |         时长        | 
| --------- | ------------------ | ---------------- | ----- | ----- | --------------- | --------- | ------------------- |
| True     |        False       |        False     | 1     | 2048  |       True      |   True    | 4分钟14秒394毫秒  |

~~~bash
!python -u ~/work/PaddleRec/tools/paddle_infer.py --model_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdmodel --params_file=/home/aistudio/output_model_all_deepfm/0/tostatic.pdiparams --use_gpu=False --data_dir=/home/aistudio/work/PaddleRec/datasets/criteo/slot_test_data_full --reader_file=/home/aistudio/work/PaddleRec/models/rank/deepfm/criteo_reader.py --batchsize=2048
~~~
日志：
~~~bash
[2021/12/15 17:39:05] root INFO: 
[2021/12/15 17:39:05] root INFO: ---------------------- Env info ----------------------
[2021/12/15 17:39:05] root INFO:  OS_version: Ubuntu 16.04
[2021/12/15 17:39:05] root INFO:  CUDA_version: 10.1.243
[2021/12/15 17:39:05] root INFO:  CUDNN_version: 7.3.1
[2021/12/15 17:39:05] root INFO:  drivier_version: 418.67
[2021/12/15 17:39:05] root INFO: ---------------------- Paddle info ----------------------
[2021/12/15 17:39:05] root INFO:  paddle_version: 2.2.1
[2021/12/15 17:39:05] root INFO:  paddle_commit: a5cf2e305b744e3ebd2f2210341f88d349d4ec5e
[2021/12/15 17:39:05] root INFO:  log_api_version: 1.0
[2021/12/15 17:39:05] root INFO: ----------------------- Conf info -----------------------
[2021/12/15 17:39:05] root INFO:  runtime_device: gpu
[2021/12/15 17:39:05] root INFO:  ir_optim: True
[2021/12/15 17:39:05] root INFO:  enable_memory_optim: True
[2021/12/15 17:39:05] root INFO:  enable_tensorrt: False
[2021/12/15 17:39:05] root INFO:  enable_mkldnn: False
[2021/12/15 17:39:05] root INFO:  cpu_math_library_num_threads: 1
[2021/12/15 17:39:05] root INFO: ----------------------- Model info ----------------------
[2021/12/15 17:39:05] root INFO:  model_name: rec_model
[2021/12/15 17:39:05] root INFO:  precision: None
[2021/12/15 17:39:05] root INFO: ----------------------- Data info -----------------------
[2021/12/15 17:39:05] root INFO:  batch_size: 2048
[2021/12/15 17:39:05] root INFO:  input_shape: dynamic
[2021/12/15 17:39:05] root INFO:  data_num: 898
[2021/12/15 17:39:05] root INFO: ----------------------- Perf info -----------------------
[2021/12/15 17:39:05] root INFO:  cpu_rss(MB): 2390.4297, gpu_rss(MB): 751.0, gpu_util: 6.0%
[2021/12/15 17:39:05] root INFO:  total time spent(s): 5.5889
[2021/12/15 17:39:05] root INFO:  preprocess_time(ms): 3.8735, inference_time(ms): 2.3315, postprocess_time(ms): 0.0187
~~~