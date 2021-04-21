[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMEME-Phoenix%2FMEME&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)   

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMEME-Phoenix%2FMEME&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="./logo.png" alt="Markdownify" width="400"></a>
  <br>
  Autonomous Driving Trolley, MEME
  <br>
</h1>

<h4 align="center">당신의 쇼핑 생활을 upgrade해 줄 <a href="https://www.notion.so/Autonomous-Driving-Trolley-MEME-01fdd602990b4baa9b603d419a1479bb" target="_blank">MEME</a>.</h4>

<p align="center">
  <a href="## ✨ Project Overview">Project Overview</a> •
  <a href="## 🛎️ How To Use">How To Use</a> •
  <a href="## 👥 Contributors & Blogs">Contributors & Blogs</a> •
  <a href="## 🔭 References">References</a> •
  <a href="## 🏆 Awards">Aq</a> •
  <a href="#license">License</a> 
</p>

<p align="center">
  <a href="https://youtu.be/IFA_xfMm73E">Demo</a> •
  <a href="https://bit.ly/3lN3iEF">Notion</a> •
  <a href="https://github.com/MEME-Phoenix">GitHub</a>
</p>

## ✨ Project Overview

### ‼️ Key Features

1. Object Tracking System with YOLOv5 & DeepSORT
2. Keyword Spotting: RNN model on word "미미야"
3. Realtime Location Track with LiDAR Sensor
4. Emergency Detection with ultrasonic Sensor
5. Embedding System

### ⚒️ System Architecture

<p align="center" >
<img src="./system_architecture_eng.png" alt="Markdownify" width="800" >
</p>

## 🛎️ How To Use

Our project is based on Ubuntu 18.04.

- Prerequisites
  To clone and run this application, you'll need [Git](https://git-scm.com) installed on your computer. From your command line:

### 1. Object Tracking System with YOLOv5 & DeepSORT

- Requirements
  Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7. To install run:
  `pip install -U -r requirements.txt`

```bash
# Clone this repository
$ git clone https://github.com/MEME-Phoenix/MEME.git

# Just run the file
$ python3 track.py
```

Note: If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/) or use `node` from the command prompt.

### 2. Keyword Spotting: RNN model on word "미미야"

Jupyter Notebook, Python versions are all set!

```bash
# Clone this repository
$ git clone https://github.com/MEME-Phoenix/MEME.git

# Go to file
$ cd keyword_spotting

# Run file
$ python3 train.py      #want to train model
$ python3 real_time.py  #use real time model
```

## 👥 Contributors & Blogs

- 한지수 [@JisuHann](https://github.com/JisuHann)
  - 스타트 학기
    - [[졸업프로젝트 개요, 1탄 RNN] 딥러닝을 이용한 자율주행카트](https://jisuhan.tistory.com/entry/졸업프로젝트딥러닝을-이용한-자율주행카트)
    - [[졸업프로젝트 2탄, CNN] ResNet50 톺아보기: 구조와 코드 분석](https://jisuhan.tistory.com/entry/CNN-ResNet50-톺아보기-구조와-코드-분석)
    - [[졸업프로젝트 3탄, HW] turtlebot3로 SLAM, Navigation 구현(2020 Summer)](https://jisuhan.tistory.com/entry/turtlebot3로-SLAM-Navigation-구현하기)
  - 그로쓰 학기
- 박지윤 [@jiyoonpark0207](https://github.com/jiyoonpark0207)
  - 스타트 학기
    - [[1탄] Yolo v3 를 이용한 인물 추적 프로젝트](https://yumissfortune.tistory.com/4)
    - [[2탄] Yolo v3 를 이용한 인물 추적 프로젝트](https://yumissfortune.tistory.com/5)
  - 그로쓰 학기
- 김채원 [@cwkim0314](https://github.com/cwkim0314)
  - 스타트 학기
    - [[IT/KR/Project] 자율 주행 카트를 만들어보자](https://blog.naver.com/cwkim0314/222156573981)
    - [[IT/KR] Object Detection - EfficientDet](https://blog.naver.com/cwkim0314/222156584109)
    - [[IT/KR/Project] Hardware: Alphabot2-pi](https://blog.naver.com/cwkim0314/222167401417)
  - 그로쓰 학기

## 🔭 References

1. Object Tracking
   - Object Tracking(https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
   - Simple Online and Realtime Tracking with a Deep Association Metric(https://arxiv.org/abs/1703.07402)
   - YOLOv4: Optimal Speed and Accuracy of Object Detection(https://arxiv.org/pdf/2004.10934.pdf)
2. Keyword Spotting
3. Embedding System

- Alpha-Bot()

5.

## 🏆 Awards

- 2020 캡스톤경진대회 창업아이디어 부문 우수상

## License

Copyright (c) 2021 MEME-Phoenix See the file license.txt for copying permission. LICENSE를 참고하세요.
