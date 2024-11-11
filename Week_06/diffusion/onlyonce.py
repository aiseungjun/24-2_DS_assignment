import os
import pickle
from PIL import Image
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# CIFAR-10 데이터셋 경로
cifar10_path = './cifar-10-batches-py'
output_path = './cifar10_train'

# 클래스 폴더 생성
os.makedirs(output_path, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(output_path, str(i)), exist_ok=True)

# 데이터 파일 불러오기
for i in range(1, 6):
    data_batch = unpickle(os.path.join(cifar10_path, f'data_batch_{i}'))
    images = data_batch['data']
    labels = data_batch['labels']

    for j in range(len(images)):
        img = images[j].reshape(3, 32, 32).transpose(1, 2, 0)  # CHW to HWC
        img = Image.fromarray(np.uint8(img))  # NumPy array를 이미지로 변환
        label = labels[j]
        img.save(os.path.join(output_path, str(label), f'image_{i * 10000 + j}.png'))

print('CIFAR-10 데이터가 ./cifar10_train에 저장되었습니다.')
