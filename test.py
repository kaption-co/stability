# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import os



model_inputs = {'prompt': 'what do you think is [MASK]?'}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())

import train
