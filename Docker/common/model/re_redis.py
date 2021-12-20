from common.config.ini import *
from common.lib.dbcontrol import redisdb

def ReRedis(output_data, data_list, debug=False, user_id=None):
    '''
    参数: 
        output_data: 用于处理数据类型
        data_list: 计算结果
        debug: 打印数据
        user_id: 用户id
    返回:
        status: 状态
    '''
    status = "ok"

    try:
        r = redisdb()
        # get type
        dtype = str(output_data.dtype)
        for i in data_list:
            data = i.append(dtype)
            r.lpush("deepfm_re:"+ user_id, data)
            # 输出内容
            if debug:
                print(data)
        # 输出个数
        if debug:
            print(r.llen("deepfm_re:"+ user_id))
    except Exception as a:
        if debug:
            print(a)
        status = "error"
    return status
