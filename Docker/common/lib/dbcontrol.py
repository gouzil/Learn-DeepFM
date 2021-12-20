import redis
from common.config.ini import * 

def redisdb():
    r = redis.StrictRedis(host=RHOST , port=RPORT, db=RNUM, password=RPWD, decode_responses=True)
    return r
