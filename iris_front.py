import requests

new_measurement = {
    "sepal_length": 1.0,
    "sepal_width": 1.0,
    "petal_length": 1.0,
    "petal_width": 1.0
}


response = requests.post('http://127.0.0.1:8000/predict', json=new_measurement)
print(response.content)
getresponse = requests.post('http://127.0.0.1:8000/predict')
print(getresponse.content)
