import sys
import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import dlib
import utils
import hopenet
from skimage import io
import json

import socket
import struct
import pickle

device_id = 0
if torch.cuda.is_available():
    # 让 PyTorch 自动选择可用的 GPU 设备
    device = torch.device('cuda',device_id)
else:
    # 如果没有可用的 GPU 设备，使用 CPU
    device = torch.device('cpu')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    # parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
    #                     default=2, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default='hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
                        default='mmod_human_face_detector.dat', type=str)
    parser.add_argument('--host', dest='host', help='Host IP address to bind the socket server.',
                        default='', type=str)
    parser.add_argument('--port', dest='port', help='Port number to bind the socket server.',
                        default=8080, type=int)#9001
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    batch_size = 1
    #gpu = args.gpu_id
    snapshot_path = args.snapshot

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model = model.to(device)
    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    print('Loading snapshot.')
    # 在加载模型参数之前清空显存
    torch.cuda.empty_cache()
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path,map_location='cuda:{}'.format(device_id))
    model.load_state_dict(saved_state_dict)

    model = model.to(device)

    print('Loading data.')

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.cuda(device)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)

    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)
    print('Socket server started on {}:{}'.format(args.host, args.port))

    conn, addr = server_socket.accept()
    print('Connected by', addr)

    received_frames = 0  # 记录接收到的视频帧数量

    while True:
        # Receive the video frame from the client
        length_data = b''
        while len(length_data) < 4:
            data = conn.recv(4 - len(length_data))
            if not data:
                break
            length_data += data

        if len(length_data) == 4:
            length = struct.unpack('!I', length_data)[0]
            print("Received data length:", length)

            frame_data = b''
            while len(frame_data) < length:
                data = conn.recv(4096)
                frame_data += data
            print("Received frame length:", len(frame_data))

            frame = pickle.loads(frame_data)
            # 验证
            test = pickle.loads(frame_data)
            if test is None:
                break

            received_frames += 1
            print("Received frames:", received_frames)
        else:
            break

        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            if conf > 1.0:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max)
                y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                img = Image.fromarray(img)

                # Transform

                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(device)

                yaw, pitch, roll = model(img)

                yaw_predicted = F.softmax(yaw, dim=1)
                pitch_predicted = F.softmax(pitch, dim=1)
                roll_predicted = F.softmax(roll, dim=1)
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted =torch.sum(pitch_predicted.data[0] * idx_tensor) *3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                # 保留小数点后三位
                yaw_predicted = round(yaw_predicted.item(), 3)
                pitch_predicted = round(pitch_predicted.item(), 3)
                roll_predicted = round(roll_predicted.item(), 3)

                # 在处理得到坐标后添加以下代码
                print("Yaw:", yaw_predicted)
                print("Pitch:", pitch_predicted)
                print("Roll:", roll_predicted)

                # Send back the coordinates
                coordinates = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
                response = {'yaw': yaw_predicted, 'pitch': pitch_predicted, 'roll': roll_predicted, 'coordinates': coordinates}
                response_data = pickle.dumps(response)
                response_length = struct.pack('!Q', len(response_data))

                print("Received frames:", received_frames)
                print("Coordinates:", coordinates)
                
                conn.sendall(response_length)
                conn.sendall(response_data)

    conn.close()
    server_socket.close()