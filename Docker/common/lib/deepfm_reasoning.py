import os
import sys
import paddle
from common.config.ini import *
from common.model.re_json import ReJson
from common.model.re_redis import ReRedis
from importlib import import_module
from paddle.io import DistributedBatchSampler, DataLoader
from paddle.inference import Config
from paddle.inference import create_predictor


# 预测指标
def init_predictor():
    config = Config(model_file, params_file)
    # 是否启用GPU
    if use_gpu:
        config.enable_use_gpu(1000, 0)
        # 是否开启加速
        if enable_tensorRT:
            config.enable_tensorrt_engine(
                max_batch_size=batchsize,
                min_subgraph_size=9,
                precision_mode=paddle.inference.PrecisionType.Float32)
    else:
        config.disable_gpu()
        config.delete_pass("repeated_fc_relu_fuse_pass")
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            config.enable_mkldnn()
    predictor = create_predictor(config)
    return predictor, config

def create_data_loader(place, data_dir):
    '''
    参数:
        place(paddle.fluid.core_avx.CPUPlace): 用于计算的硬件
        place(paddle.fluid.core_avx.CUDAPlace): 用于计算的硬件
    使用CUDAPlace要指出GPU ID
    '''
    # global reader_file
    # global batchsize
    reader_path, reader_file_temp = os.path.split(reader_file)
    reader_file_temp, extension = os.path.splitext(reader_file_temp)
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    sys.path.append(reader_path)
    #sys.path.append(os.path.abspath("."))
    reader_class = import_module(reader_file_temp)
    config = {"runner.inference": True}
    dataset = reader_class.RecDataset(file_list, config=config)
    loader = DataLoader(
        dataset, batch_size=batchsize, places=place, drop_last=True)
    return loader

def sort(debug=False, full=False, re_model="json", user_id=None, data_dir=None):
    '''
    参数:
        debug(bool): 输出日志 
        full:(bool): 返回内容模式(详细或简略)
        re_model(str): 返回格式
        user_id(str): 用户id
        data_dir(str): 文件路径
    返回:
        results_js(JSON Object): (连接状态, 数据类型, 数据)
        None: 存入redis，没有返回值
    '''
    predictor, pred_config = init_predictor()
    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    if data_dir == None:
        return "error"
    test_dataloader = create_data_loader(place, data_dir)

    if debug:
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name=model_name,
            model_precision=precision,
            batch_size=batchsize,
            data_shape="dynamic",
            save_path=save_log_path,
            inference_config=pred_config,
            pids=pid,
            process_name=None,
            gpu_ids=0,
            time_keys=[
                'preprocess_time', 'inference_time', 'postprocess_time'
            ])

    # 用于存放返回值
    data_list = []
    for batch_id, batch_data in enumerate(test_dataloader):
        name_data_pair = dict(zip(input_names, batch_data))
        if debug:
            autolog.times.start()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(name_data_pair[name].numpy())
        if debug:
            autolog.times.stamp()
        predictor.run()
        for name in output_names:
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
        # 返回结果
        results = []
        # 结果类型
        results_type = []
        if debug:
            autolog.times.stamp()
        for name in output_names:
            results_type.append(output_tensor.type())
            results.append(output_data[0])
        if debug:
            autolog.times.end(stamp=True)
        # 判断是否是详细输出
        if full:
            data_list.append(output_data.tolist())
        else :
            data_list.append(results[0].tolist())
    if debug:
        autolog.report()
    

    if re_model == "json":
        results_js = ReJson(output_data, data_list, debug, user_id)
        return results_js
    elif re_model == "redis":
        redis_status = ReRedis(output_data, data_list, debug, user_id)
        return redis_status

if __name__ == '__main__':
    user_id = 77
    sort(True, True, "json", user_id)
    # 不强制退下次会重发输出日志
    exit()