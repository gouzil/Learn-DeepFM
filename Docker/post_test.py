import requests

url = 'http://127.0.0.1:8867/upload'
files = {'file': open('./work/PaddleRec/models/rank/deepfm/data/sample_data/train/sample_train.txt', 'rb')}           
params = {"debug":"true", "user_id":"11", "full":"false","re_model":"json","save":"false"}

response = requests.post(url, params=params, files=files)
json = response.text
print(json)