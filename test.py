# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import os
import requests
from PIL import Image
import numpy as np
from pathlib import Path


def get_img_arrs(image_dir, return_lists=False):
    img_array = []
    for img in os.listdir(image_dir):
        if img.endswith('.png'):
            img_path = image_dir / img
            img = Image.open(img_path)
            arr = np.array(img)
            if return_lists:
                ls = arr.tolist()
                img_array.append(ls)
            else:
                img_array.append(arr)
    return img_array


# Load image arrs
image_dir = Path('data')
img_array = get_img_arrs(image_dir, return_lists=True)
sample_inputs = img_array


if __name__ == '__main__':
    import os
    import banana_dev as banana
    from time import time
    from dotenv import load_dotenv

    # load_dotenv()
    # api_key = os.getenv('BANANA_API_KEY')
    # print('ApiKey: ', api_key)
    # model_key = os.getenv('BANANA_MODEL_ID')
    # print('ModelKey: ', model_key)

    # for _ in range(7):
    #     sample_inputs += sample_inputs
    # print(len(sample_inputs))

    model_inputs = {
        'inputs': sample_inputs
    }

    # GPU API Ping
    # t0 = time()
    # out = banana.run(api_key, model_key, model_inputs={'ping': 'true'})
    # print('GPU API Ping: ({:.2f}s)\n'.format(time() - t0))

    # GPU API
    # t0 = time()
    # out = banana.run(api_key, model_key, model_inputs)
    # print('GPU API: ({:.2f}s) / network'.format(time() - t0))
    # print('GPU API: ({:.2f}s) / compute (pred)'.format(out['modelOutputs'][0]['time']))
    # print('GPU API: ({:.2f}s) / compute-time total'.format(out['modelOutputs'][0]['total_time']))
    # print('Device: {}'.format(out['modelOutputs'][0]['device']))
    # print('Examples: {}\n'.format(len(sample_inputs)))

    # CPU (local)
    t0 = time()
    res = requests.post('http://localhost:8000/', json=model_inputs)
    print(res.json())
    print('CPU (local): ({:.2f})'.format(time() - t0))
    print('Examples: {}\n'.format(len(img_array)))
