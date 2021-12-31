# dcn

auc_list累加值 8953.777222812176 平均值 0.7703788681825002

infer_auc累加值 1124.8016819357872 平均值 0.7580646434095171

配置文件：

```
runner:
  train_data_dir: "../../../datasets/criteo/slot_train_data_full"
  train_reader_path: "reader" # importlib format
  use_gpu: True
  use_auc: True
  use_visual: True
  train_batch_size: 512
  epochs: 10
  print_interval: 10
  #model_init_path: "output_model/0" # init model
  model_save_path: "big_model_deepCross"
  test_data_dir: "../../../datasets/criteo/slot_test_data_full"
  infer_reader_path: "reader" # importlib format
  infer_batch_size: 512
  infer_load_path: "big_model_deepCross"
  infer_start_epoch: 0
  infer_end_epoch: 10


# hyper parameters of user-defined network
hyper_parameters:
  # optimizer config
  optimizer:
    class: Adam
    learning_rate: 0.0001
    strategy: async
  # user-defined <key, value> pairs
  sparse_inputs_slots: 27
  sparse_feature_number: 1000001
  sparse_feature_dim: 9
  dense_input_dim: 13
  fc_sizes: [512, 256, 128] #, 32]
  distributed_embedding: 0

# sparse_inputs_slots + dense_input_dim

  cross_num: 2
  l2_reg_cross: 0.00005
  dnn_use_bn: False
  clip_by_norm: 100.0
  is_sparse: False
  # cat_feat_num: "{workspace}/data/sample_data/cat_feature_num.txt"
```