import numpy as np
import cv2
from tqdm.auto import trange
import math
import heapq
import numpy as np
import os
#here
import tensorflow as tf
graph = tf.get_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
from PIL import Image
from core.utils import load_image, deprocess_image, preprocess_image
from core.networks import unet_spp_large_swish_generator_model
from core.dcp import estimate_transmission
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
import yolov5.detect
import shutil

img_size = 512


def preprocess_image(cv_img):
    cv_img = cv2.resize(cv_img, (img_size, img_size))
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def load_image(path):
    img = Image.open(path)
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


def preprocess_cv2_image(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, (img_size, img_size))
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def preprocess_depth_img(cv_img):
    cv_img = cv2.resize(cv_img, (img_size, img_size))
    img = np.array(cv_img)
    img = np.reshape(img, (img_size, img_size, 1))
    img = 2 * (img - 0.5)
    return img


g = unet_spp_large_swish_generator_model()
weight_path = "./weights/densehaze_generator_in512_ep85_loss227.h5"
g.load_weights(weight_path)
g.summary()


def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex


def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex


variance_list = [15, 80, 30]


# ( 연기를 제거하려는 원본 영상의 경로[해당 영상을 불러 온다], 최종 결과물이 저장되는 경로, 파일 이름(확장자포함) )
def desmoke_video(video_path, path1, result_name):
    video = cv2.VideoCapture(video_path)
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (width, height)
    out = cv2.VideoWriter(path1, cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=size)
    for i in trange(count):
        ret, frame = video.read()
        if ret:
            Frame = frame
            # Frame = fdehz(frame)
            # Frame = np.invert(frame)
            # Frame = dehazing(frame)
            # Frame = np.invert(Frame)
            # Frame = dehazing(Frame)
            # Frame = fdehz(Frame)
            # Frame = MSR(frame,variance_list)
            h, w, _ = Frame.shape
            t = estimate_transmission(Frame)
            t = preprocess_depth_img(t)
            Frame = preprocess_cv2_image(Frame)
            x_test = np.concatenate((Frame, t), axis=2)
            x_test = np.reshape(x_test, (1, img_size, img_size, 4))
            with graph.as_default():
                generated_images = g.predict(x=x_test)
            #generated_images = g.predict(x=x_test)
            de_test = deprocess_image(generated_images)
            de_test = np.reshape(de_test, (img_size, img_size, 3))
            de_test = cv2.resize(de_test, (w, h))
            rgb_de_test = cv2.cvtColor(de_test, cv2.COLOR_BGR2RGB)
            out.write(rgb_de_test)
    video.release()
    out.release()
    # 영상 처리 완료

    yolov5.detect.run(weights='./yolov5/yolov5s.pt', source=path1, imgsz=size)

    # 욜로 적용된 다음 저장되는 위치
    shutil.move("./yolov5/runs/detect/exp/" + result_name, path1)

    if os.path.exists("./yolov5/runs/detect/exp"):
        shutil.rmtree("./yolov5/runs/detect/exp")

# desmoke_video() 사용법
# desmoke_video(연기 제거하려는 영상 경로, 영상처리 결과를 저장할 경로, 영상처리 결과를 저장할 경로 중 파일 이름만)
# desmoke_video("./test_video/야외1.mp4", "./result_video/processed_야외1_test_for_yolo.mp4", "processed_야외1_test_for_yolo.mp4")
