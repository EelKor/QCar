from numpy.core.defchararray import endswith
from Quanser.q_essential import Camera2D
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
simulationTime = 100.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
counter = 0
imageWidth = 1280   
imageHeight = 720
cameraID = '2'

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the CSI cameras
myCam = Camera2D(camera_id=cameraID, frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Main Loop
try:
    def mouse_callback(event, x,y,flags, param):
        print(x,y)
    #cv2.setMouseCallback('Test Gray', mouse_callback)

    while elapsed_time() < simulationTime:
        
        # Start timing this iteration
        start = time.time()

        # Capture RGB Image from CSI
        myCam.read()
        counter += 1

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # ########## OPEN CV ############### #
        # 화면 마우스를 이용해 좌표값 추출 
        #cv2.setMouseCallback('Test Gray', mouse_callback)

        # 캐니 값 조정 변수
        thr1 = 10
        thr2 = 50

        # 탑 뷰 이미지 변환 행렬
        srcPoint = np.array([[198,310],[433,307],[604,375],[27,386]], dtype=np.float32)
        dsrPoint = np.array([[0,0], [640,0],[640,480],[0,480]], dtype=np.float32)
        topViewMatrix = cv2.getPerspectiveTransform(srcPoint, dsrPoint)

        # 흑백 이미지 생성
        imGray = cv2.cvtColor(myCam.image_data, cv2.COLOR_BGR2GRAY)
        # Top View 이미지 생성
        dst = cv2.warpPerspective(imGray, topViewMatrix, (640,480))
        # Canny 이미지 생성
        canny = cv2.Canny(dst, thr1, thr2)

        # 이미지 표시
        #cv2.imshow('Test thr', canny)
        cv2.imshow('Test Gray', imGray)
        cv2.imshow('Top View', dst)

        
        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
        cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Terminate all webcam objects    
    myCam.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
