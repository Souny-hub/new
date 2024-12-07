import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 加载YOLOv11模型（使用GPU时自动启用）
model = YOLO("best.pt")

# 设置Streamlit页面的标题和描述
st.header('YOLO11 Real-Time Object Detection')
st.subheader('-'*60)
st.write('公众号：Souny的深度学习')

# 创建一个视频流捕获对象
cap = cv2.VideoCapture(0)  # 也可以替换为网络摄像头的URL

if not cap.isOpened():
    st.error("无法打开摄像头，请检查连接或配置。")
else:
    # 实时推理视频流
    stframe = st.empty()  # 用于实时显示视频帧

    frame_counter = 0  # 用于控制每秒处理的帧数
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 降低帧率：每3帧处理一次
        frame_counter += 1
        if frame_counter % 3 != 0:
            continue

        # 调整图像分辨率（减小图像尺寸，提高推理速度）
        small_frame = cv2.resize(frame, (640, 480))  # 调整分辨率到 640x480

        # 使用YOLOv11模型进行推理
        results = model(small_frame)

        # 获取带有检测框的结果图像
        result_frame = results[0].plot()  # 绘制检测框

        # 将结果帧从BGR转换为RGB（因为OpenCV默认BGR，而Streamlit需要RGB）
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # 使用Streamlit展示实时视频帧
        stframe.image(result_frame_rgb, channels="RGB", use_column_width=True)

    cap.release()  # 释放摄像头资源
