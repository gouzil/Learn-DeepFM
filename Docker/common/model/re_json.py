import json

def ReJson(output_data, data_list, debug=False, user_id=None):
    '''
    参数: 
        output_data: 用于处理数据类型
        data_list: 计算结果
        debug: 打印数据
        user_id: 用户id
    返回：
        results_js(JSON Object): (连接状态, 数据类型, 数据)
    '''
    try:
        # 构建返回json
        results_js = {}
        results_js["code"] = 200
        results_js["dtype"] = str(output_data.dtype)
        results_js["data"] = data_list
        results_js["user_id"] = user_id
        results_js = json.dumps(results_js)
        if debug:
            print(results_js)
        return results_js
    except Exception as a:
        results_js["code"] = 401
        results_js["data"] = a
        results_js = json.dumps(results_js)
        if debug:
            print(a)
        return results_js
