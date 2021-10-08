from math import atan
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
from Quanser.q_interpretation import *
from Quanser.q_control import *
from Quanser.q_dp import *
from Quanser.q_misc import *
from Quanser.q_ui import *
from Quanser.q_essential import *

import time
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
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)

# Initialize motor command array
mtr_cmd = np.array([0,0])

# AutoPilot Activated
isAutoPilotOn = False

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## 필터 세팅 - 조향 관련
steering_filter = Filter().low_pass_first_order_variable(25, 0.033)
next(steering_filter)
dt = 0.033

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## QCar 하드웨에 제어 초기화
## 조종기 초기화 - gamepadViaTarget(x+1), x = 조종기id , ls -l /dev/input/by-id 에서 exent5 이런식으로 숫자
myCar = QCar()
gpad = gamepadViaTarget(6)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Driving Configuration: Use 3 toggles or 4 toggles mode as you see fit:
# Common to both 3 or 4 mode
#   Steering                    - Left Lateral axis
#   Arm                         - LB
# In 3 mode: 
#   Throttle (Drive or Reverse) - Right Longitudonal axis
# In 4 mode:
#   Throttle                    - Right Trigger (always positive)
#   Button A                    - Reverse if held, Drive otherwise
configuration = '3' # change to '4' if required


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the RealSense camera for RGB and Depth data
myCam1 = Camera3D(mode='RGB&DEPTH', frame_width_RGB=imageWidth, frame_height_RGB=imageHeight)
counter = 0
max_distance = 2 # maximum depth distance
                 # distances beyond will display as white pixels 

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Main Loop
try:
    # 마우스로 좌표 추출
    def mouse_callback(event, x,y,flags, param):
        print(x,y)


    # 메인 프로그램
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = time.time()

        # -------------------------------------------------------------------------------------
        ## 데이터 Acquire
        # Capture RGB and depth data
        myCam1.read_RGB()
        myCam1.read_depth(dataMode='m') # for data in meters... 
        # myCam1.read_depth(dataMode='px') # for data in pixel range 0-255

        # 조종기 데이터 수신
        new = gpad.read()

        # 루프 카운트 + 1
        counter += 1
        
        # --------------------------------------------------------------------------------------
        ## 이미지 프로세스 부분

        # 마우스 좌표 반환 콜백함수
        cv2.setMouseCallback('result', mouse_callback)

        # 흑백 영상 생성 및 가우시안 블러로 노이즈 필터링
        imgray = cv2.cvtColor(myCam1.image_buffer_RGB, cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (0,0), 2)
        imDepth = cv2.GaussianBlur(myCam1.image_buffer_depth_m/max_distance, (0,0), 3)

        # Top View 이미지 생성
        srcPoint = np.array([[419,486],[896,477],[1274,696],[17,711]], dtype=np.float32)
        dsrPoint = np.array([[0,0], [imageWidth,0],[imageWidth,imageHeight],[0,imageHeight]], dtype=np.float32)
        topViewMatrix = cv2.getPerspectiveTransform(srcPoint, dsrPoint)
        dst = cv2.warpPerspective(imgray, topViewMatrix, (imageWidth,imageHeight))

        #바이너리 이미지 생성
        ret, binary = cv2.threshold(dst, 170, 255, cv2.THRESH_BINARY )
        # 캐니 이미지 생성 및 결과창 정의
        canny = cv2.Canny(imgray,20, 170)
        result = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
        # ROI 지정
        canny_ROI = canny[469:720, 0:1280]

        # ------------------------------------------------------------------------------------
        ## 확률허프 변환 직선 감지
        avg_slop_theta = 0
        lineP = cv2.HoughLinesP(canny_ROI, 1, np.pi / 180, 10, None, 10, 20)

        # 감지된 직선의 기울기 계산
        if lineP is not None:
            for j in range(0, len(lineP)):
                l = lineP[j][0]
                dx = l[2]-l[0]
                dy = l[3]-l[1]
                avg_slop_theta += (dx/dy)
            avg_slop_theta /= len(lineP)
            avg_slop_theta = atan(avg_slop_theta) 
                
        if lineP is not None:
            for i in range(0, len(lineP)):
                l = lineP[i][0]
                cv2.line(result, (l[0],l[1]+469), (l[2],l[3]+469), COLOR_RED, 2, cv2.LINE_AA)

        cv2.line(result,(640,360),(int(640+200*np.cos(avg_slop_theta)),int(360+200*np.sin(avg_slop_theta))), COLOR_GREEN, 2, cv2.LINE_AA )
        # -----------------------------------------------------------------------------------
        ## 조향값 계산

        # 감지된 라인의 기울기 및 y절편 구하기
        slope, intercept = find_slope_intercept_from_binary(binary)

        #조향값 계산
        raw_steering = 1.5*(slope + 0.02) + (1/150)*(intercept-366)
        steering = steering_filter.send((saturate(raw_steering, 0.5, -0.5), dt))

        # ------------------------------------------------------------------------------------
        ## 오토파일럿 및 수동조종

        # 오토파일럿 기능 활성화 확인 X 버튼 으로 On/off
        if gpad.X:
            isAutoPilotOn = not isAutoPilotOn

        # 오토파일럿 모드
        if isAutoPilotOn:
            # 오토파일럿 해제 조건
            if gpad.LB and new:
                isAutoPilotOn = False
                mtr_cmd = np.array([0.01*gpad.RLO, 0.5*gpad.LLA]) 
                
            else:
                # 스로틀 고정 및 자동 스티어링
                mtr_cmd = np.array([0.07, raw_steering])

        # 수동 조종 모드
        else:
            if new and gpad.LB:
                mtr_cmd = np.array([0.3*gpad.RLO, 0.5*gpad.LLA]) 
 
        # 하드웨어 제어 명령 전송
        myCar.write_mtrs(mtr_cmd)

        # ------------------------------------------------------------------------------------
        # End timing this iteration

        end = time.time()

        # Calculate the computation time, and the time  that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # -------------------------------------------------------------------------------------
        #  Display the two images

        cv2.imshow('result', result)
        #cv2.imshow('Top View', dst)
        #cv2.imshow("Canny", canny)
        cv2.imshow('ROI',canny_ROI)
        #cv2.imshow('Binary', binary)

        #cv2.imshow('My Depth', imDepth) # by default, ranges between 0 to 1 meters will show up in grayscale. Beyond 1 m, everything will look white. 
                                                            # apply a gain of 0.5 to improve range to 2 meters, 0.33 for 3 meters and so on
        # cv2.imshow('My Depth', 30*myCam1.image_buffer_depth_px) # apply a gain of 30 to visualize data if using pixels

        #------------------------------------------------------------------------------------
        # 디버그 데이터 값 출력
        
        #if lineP is not None:
        #    print(lineP[0][0], lineP[1][0], avg_slop_theta*180/np.pi)

        #print("slope: ", slope, "inter: ", intercept)
        #print("Raw Steering ", raw_steering)
        #print("steering ", steering)
        #print(computationTime)

        if isAutoPilotOn:
            # 결과 보고
            print("[AutoPilot]\t\tThrottle:{0:1.2f} Steering: {1:1.2f}"
                                            .format(mtr_cmd[0], mtr_cmd[1]))
        elif not isAutoPilotOn:
            # 결과 보고
            print("[Manual]\t\tThrottle:{0:1.2f} Steering: {1:1.2f}"
                                            .format(mtr_cmd[0], mtr_cmd[1]))



        # --------------------------------------------------------------------------------------
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
