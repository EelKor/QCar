from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
from Quanser.q_interpretation import *
from Quanser.q_control import *
from Quanser.q_dp import *
from Quanser.q_misc import *
from Quanser.q_ui import *
from Quanser.q_essential import *

import time
import struct
import numpy as np 
import cv2

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 30.0
sampleTime = 1/sampleRate
simulationTime = 200.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
counter = 0
imageWidth = 1280
imageHeight = 720

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## 필터 세팅 - 조향 관련
steering_filter = Filter().low_pass_first_order_variable(25, 0.033)
next(steering_filter)
dt = 0.033

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## QCar 하드웨에 제어 초기화
myCar = QCar()

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the RealSense camera for RGB and Depth data
myCam1 = Camera3D(mode='RGB&DEPTH', frame_width_RGB=imageWidth, frame_height_RGB=imageHeight)
counter = 0
max_distance = 5 # maximum depth distance
                 # distances beyond will display as white pixels 

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Main Loop
try:
    # 마우스로 좌표 추출
    def mouse_callback(event, x,y,flags, param):
        print(x,y)
    #cv2.setMouseCallback('Test Gray', mouse_callback)


    # 메인 프로그램
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = time.time()

        # Capture RGB and depth data
        myCam1.read_RGB()
        myCam1.read_depth(dataMode='m') # for data in meters... 
        # myCam1.read_depth(dataMode='px') # for data in pixel range 0-255

        counter += 1
        
        #################################################
        # 이미지 프로세스 부분

        cv2.setMouseCallback('Test Gray', mouse_callback)

        srcPoint = np.array([[419,486],[896,477],[1274,696],[17,711]], dtype=np.float32)
        dsrPoint = np.array([[0,0], [imageWidth,0],[imageWidth,imageHeight],[0,imageHeight]], dtype=np.float32)
        topViewMatrix = cv2.getPerspectiveTransform(srcPoint, dsrPoint)

        imgray = cv2.cvtColor(myCam1.image_buffer_RGB, cv2.COLOR_BGR2GRAY)
        # Top View 이미지 생성
        dst = cv2.warpPerspective(imgray, topViewMatrix, (imageWidth,imageHeight))

        #바이너리 이미지 생성
        ret, binary = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY )

        # 감지된 라인의 기울기 및 y절편 구하기
        slope, intercept = find_slope_intercept_from_binary(binary)

        # 조향값 계산
        raw_steering = 0.5*(slope + 0.02) + (1/300)*(intercept-366)
        steering = steering_filter.send((saturate(raw_steering, 0.5, -0.5), dt))
        print("slope: ", slope, "inter: ", intercept)
        print("Raw Steering ", raw_steering)
        print("steering ", steering)
        # 조향 명령 전송
        mtr_cmd = np.array([0.06,0])
        mtr_cmd[1] = raw_steering
        myCar.write_mtrs(mtr_cmd)
        ####################################################
        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time  that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # Display the two images
        cv2.imshow('Top View', dst)
        cv2.imshow('Binary', binary)
        #cv2.imshow('My Depth', myCam1.image_buffer_depth_m/max_distance) # by default, ranges between 0 to 1 meters will show up in grayscale. Beyond 1 m, everything will look white. 
                                                            # apply a gain of 0.5 to improve range to 2 meters, 0.33 for 3 meters and so on
        # cv2.imshow('My Depth', 30*myCam1.image_buffer_depth_px) # apply a gain of 30 to visualize data if using pixels

        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
    print("User interrupted!")

finally:    
    # Terminate RealSense camera object
    myCam1.terminate()
    myCar.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
