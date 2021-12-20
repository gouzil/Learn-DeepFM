import configparser

try:
    # 读取config文件
    cf = configparser.ConfigParser()
    cf.read("./config.ini", encoding='utf-8-sig')

    # deepfm
    use_gpu = cf.get("DEEPFM", "USE_GPU")
    if use_gpu == "True" or use_gpu == "true":
        use_gpu = True
    elif use_gpu == "False" or use_gpu == "false":
        use_gpu =False

    model_file = cf.get("DEEPFM", "MODEL_FILE")
    params_file = cf.get("DEEPFM", "PARAMS_FILE")
    # 改由post传入
    # data_dir = cf.get("DEEPFM", "DATA_DIR")
    reader_file = cf.get("DEEPFM", "READER_FILE")
    batchsize = int(cf.get("DEEPFM", "BATCHSIZE"))

    enable_tensorRT = cf.get("DEEPFM", "ENABLE_TENSORRT")
    if enable_tensorRT == "True" or enable_tensorRT == "true":
        enable_tensorRT = True
    elif enable_tensorRT == "False" or enable_tensorRT == "false":
        enable_tensorRT =False

    enable_mkldnn = cf.get("DEEPFM", "ENABLE_MKLDNN")
    if enable_mkldnn == "True" or enable_mkldnn == "true":
        enable_mkldnn = True
    elif enable_mkldnn == "False" or enable_mkldnn == "false":
        enable_mkldnn =False

    cpu_threads = int(cf.get("DEEPFM", "CPU_THREADS"))
    model_name = cf.get("DEEPFM", "MODEL_NAME")
    precision = cf.get("DEEPFM", "PRECISION")
    save_log_path = cf.get("DEEPFM", "SAVE_LOG_PATH")

    # redis
    RHOST = cf.get("REDIS", "RHOST")
    RPORT = cf.get("REDIS", "RPORT")
    RNUM = cf.get("REDIS", "RNUM")
    RPWD = cf.get("REDIS", "RPWD")

    # falsk
    FHOST = cf.get("FLASK", "FHOST")
    FPORT = cf.get("FLASK", "FPORT")
    FDEBUG = cf.get("FLASK", "FDEBUG")

except Exception as a:
    print(a)