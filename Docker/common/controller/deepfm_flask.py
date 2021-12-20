import os
import time
import shutil
import random
from flask import Flask, request
from werkzeug.utils import secure_filename
from common.config.ini import *
from common.lib.deepfm_reasoning import sort


app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None:
            # 表示没有发送文件
            return "未上传文件"
        # 加入随机数避免重复
        rdmt = str(time.time()) + str(random.randint(0,1000))
        rdmt = rdmt.replace('.','')
        data_dir = f"deepfm/data/{rdmt + secure_filename(file.filename)}"
        os.makedirs(data_dir)
        file.save(data_dir + "/" + f"{secure_filename(file.filename)}")
    
    # 做预处理(鬼知道会传什么神奇的东西上来)
    # debug模式
    debug = request.args.get("debug")
    if debug == "True" or debug == "true":
        debug = True
    elif debug == "False" or debug == "false":
        debug = False
    else:
        debug = False
    # print(debug)

    # 用户id
    user_id = request.args.get("user_id")
    if user_id == None:
        return "请上传用户id"
    # print(user_id)

    # 返回内容模式(详细或简略)
    full = request.args.get("full")
    if full == "True" or full == "true":
        full = True
    elif full == "False" or full == "false":
        full = False
    else:
        full = False
    # print(full)

    # 返回格式
    re_model = request.args.get("re_model")
    if re_model == "json" or re_model == "Json":
        re_model = "json"
    elif re_model == "redis" or re_model == "Redis":
        re_model = "redis"
    else:
        re_model = "json"
    # print(re_model)

    results = sort(debug, full, re_model, user_id, data_dir)

    # 是否保存
    save = request.args.get("save")
    if save == "True" or save == "true":
        pass
    elif save == "False" or save == "false":
        shutil.rmtree(data_dir)
    else:
        shutil.rmtree(data_dir)
    # print(save)

    return results

def main(local=False):
    if local:
        app.run(host="0.0.0.0", port=8867,debug=True)
    else:
        app.run(host=FHOST, port=FPORT,debug=FDEBUG)

if __name__ == '__main__':
    main(True)