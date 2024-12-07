import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# 加载YOLOv11模型（使用GPU时自动启用）
model = YOLO("best.pt")

# 自定义页面样式，模仿苹果风格
st.markdown(
    """
    <style>
        .main {
            background-color: #f9f9f9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        .css-18e3th9 {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            font-weight: 600;
            color: #333333;
        }
        .sidebar .sidebar-content {
            background-color: #f4f4f4;
            border-right: 1px solid #e6e6e6;
        }
        .block-container {
            padding: 1.5rem 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 设置页面标题和描述
st.title('🚀 实时物体检测 🚀')
st.header(" 🚀 人工智能机器视觉实验室 🚀")
st.write("""
一个使用 视觉算法 进行实时物体检测的演示平台。
""")

# 侧边栏设置
with st.sidebar:
    st.header("控制面板")
    st.write("调整参数以优化检测效果。")
    frame_interval = st.slider("帧处理间隔（单位：帧）", 1, 10, 3)
    detection_confidence = st.slider("检测置信度阈值", 0.1, 1.0, 0.5)

# 创建视频捕获对象
cap = cv2.VideoCapture(0)  # 可替换为网络摄像头的 URL
if not cap.isOpened():
    st.error("\u26a0\ufe0f 无法打开摄像头，请检查连接或配置。")
else:
    # 实时推理视频流
    stframe = st.empty()  # 用于实时显示视频帧
    frame_counter = 0
    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # 控制帧处理间隔
        if frame_counter % frame_interval != 0:
            continue

        # 调整图像分辨率（减小图像尺寸，提高推理速度）
        small_frame = cv2.resize(frame, (640, 480))

        # 使用YOLO模型进行推理
        results = model(small_frame, conf=detection_confidence)

        # 获取带有检测框的结果图像
        result_frame = results[0].plot()  # 绘制检测框

        # 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 将结果帧从BGR转换为RGB
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # 在图像上显示FPS
        cv2.putText(
            result_frame_rgb,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50, 50, 50),
            2
        )

        # 用Streamlit显示视频流
        stframe.image(result_frame_rgb, channels="RGB", use_container_width=True)

    cap.release()  # 释放摄像头资源
