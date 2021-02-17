from evdev import InputDevice, categorize, ecodes, KeyEvent
from adafruit_servokit import ServoKit
import inputs
import time
import threading
import queue
from time import sleep
import os
from os import path
import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import model_from_yaml
from tensorflow.python.keras.backend import set_session
# from merge_ver import load_model as lm
# from os import ..merge_ver.load_model
# from ..merge_ver.load_model import load_model as lm
from Hayoung.module import ultraSonic
import logging

## GLOBAL VARIABLES...
# import sys
# sys.path.append("..")

camera = []
list_set = []
count = 0
#kit,gamepad,loaded_model,q=[None,None,None,None]


config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

with tf.Session(config=config) as session:
    init = tf.compat.v1.global_variables_initializer()
    graph = tf.compat.v1.get_default_graph()

    session.run(init)
    def gstreamer_pipeline(
        capture_width=120,#1280,
        capture_height=320,#720,
        display_width=120,#1280,
        display_height=320,#720,
        framerate=15,
        flip_method=2,
    ):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (capture_width,capture_height,framerate,flip_method,display_width,display_height,))


    def load_model(path):
        yamlPath = path[0]
        h5Path = path[1]
        yaml_file1 = open(yamlPath, 'r')
        loaded_model_yaml = yaml_file1.read()
        yaml_file1.close()
        model = model_from_yaml(loaded_model_yaml)

        # load weights into new model
        model.load_weights(h5Path)
        model._make_predict_function()

        return model

    def show_camera():
        global q
        idx = 1
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            #window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
            # Window

            while 1:
                ret_val, img = cap.read() # img is input image's data
                #cv2.imshow("CSI Camera", img)
                # This also acts as
                #img = cv2.imread(img)
                img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                #print(img.shape)
                img = img.reshape(-1,320,120,1)
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
                q.put(img)

                time.sleep(0.05)

                break
            cap.release()

            #cv2.destroyAllWindows()
        else:
            print("Unable to open camera")


    def human_detect():
        global q2
        dispW = 240
        dispH = 180
        flip = 2

        # Or, if you have a WEB cam, uncomment the next line
        # (If it does not work, try setting to '1' instead of '0')
        cam = cv2.VideoCapture(1)
        cam.set(cv2.CAP_PROP_FPS, 10)
        # cam.set(3, dispW)  # 3 : width
        # cam.set(4, dispH)  # 4 : height


        # time.sleep(0.1)
        ret, frame = cam.read()

        frame = cv2.resize(frame,(180, 240))

        t1 = time.time()
        #while True:
        if time.time() - t1 > 5 :
            print('5second')
            #break


        ret, frame = cam.read()
        frame = cv2.resize(frame, (180,240))
        #print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT), cam.get(cv2.CAP_PROP_FPS))


        #print('before color convert size is ',frame.shape)
        time.sleep(0.1)
#            frame=np.reshape(frame,(-1,dispW,dispH,))

        # model v6 test preprocessing #################################################
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(frame, (5, 5), 2)
        frame = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        frame = np.reshape(frame,(-1,180,240,1))

        ################################################################################
        #print('after color convert size is ',frame.shape)
        #frame = np.reshape(frame,(-1, 180, 240,3))

        q2.put(frame)
        #cv2.imshow('nanoCam', frame)

        cam.release()
        cv2.destroyAllWindows()


    def drive_control(dir):
        if dir == 0:
            # Forward(Straight)
            kit.servo[1].angle = 107
            kit.continuous_servo[0].throttle = 0.24
            time.sleep(0.25)
            kit.continuous_servo[0].throttle = 0

            return
        elif dir ==1:
            # Right
            kit.servo[1].angle = 85
            kit.continuous_servo[0].throttle = 0.24
            time.sleep(0.3)
            kit.continuous_servo[0].throttle = 0
        elif dir ==2:
            # Left
            kit.servo[1].angle = 145
            kit.continuous_servo[0].throttle = 0.24
            time.sleep(0.3)
            kit.continuous_servo[0].throttle = 0

            return

        elif dir ==3:
            # BackWard
            kit.servo[1].angle = 107
            kit.continuous_servo[0].throttle = -0.181
            return

    def get_drive_direction(num):
        if num ==0:
            return "straight"
        elif num ==1:
            return 'turn right'

        else:
            return 'turn left'

    def predict():
        global lane_model,motion_model,q,ir, q2
        X = q.get()
        IR = ir.get()

        with graph.as_default():
            set_session(session)

            pred_raw=lane_model.predict(X)
            # if np.max(pred_raw)<0.8 : # threashold : 가장 높은 확률로 예측된 class의 확률이 50% 미만이면 go straight
            #     pred=0
            # else:
            #     pred=np.argmax(pred_raw)
            # #print("predicted value is ",np.argmax(lane_model.predict(X)))
            print(IR)
            if IR:
                pred = np.argmax(pred_raw)
                drive_control(pred)

                mylogger.info('Drive model prediction value is ' + get_drive_direction(pred))

            else:
                kit.continuous_servo[0].throttle = 0
                kit.servo[1].angle = 113
                human_detect()
                print("detection is over")
                # # 1. webcam에서 영상을 받아온다.
                while q2.empty() is not True:
                    pred_tmp = q2.get()
                    predicted_value=motion_model.predict(pred_tmp)
                    pred = np.argmax(predicted_value)

                    print("motion pred is ", pred)
                    if pred == 0: #straight
                        kit.servo[1].angle = 107
                        kit.continuous_servo[0].throttle = 0.2

                        time.sleep(0.5)
                        mylogger.info('Motion model prediction value is ' + get_drive_direction(pred))
                        mylogger.info('prob ', predicted_value)

                    elif pred == 1: #left
                        kit.servo[1].angle = 145
                        kit.continuous_servo[0].throttle = 0.2
                        time.sleep(0.5)
                        mylogger.info('Motion model prediction value is ' + get_drive_direction(pred))
                        mylogger.info('prob ', predicted_value)


                    else: #right
                        kit.servo[1].angle = 85
                        kit.continuous_servo[0].throttle = 0.2
                        time.sleep(0.5)
                        mylogger.info('Motion model prediction value is ' + get_drive_direction(pred))
                        mylogger.info('prob ', predicted_value)



    def check_dist():
        global ir

        dist = ultraSonic.distance()
        print(dist)
        if dist < 80:
        # stop
            ir.put(False)
        else:
        # keep run
            ir.put(True)

    def main():
        t0 = time.time()
        thread1 = threading.Thread(target=show_camera)
        thread2 = threading.Thread(target=predict)
        thread3 = threading.Thread(target=check_dist)

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()

        t1 = time.time()

        print("Execution Time {}".format(t1-t0))

    if __name__ == "__main__":
        kit, gamepad, q, ir, q2=[None, None,None,None,None]

        mylogger = logging.getLogger("my")
        mylogger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        mylogger.addHandler(stream_handler)

        BUF_SIZE = 4096
        q = queue.Queue(BUF_SIZE)
        q2 = queue.Queue(BUF_SIZE)
        ir = queue.Queue(BUF_SIZE)


        kit = ServoKit(channels=16)
        #gamepad =InputDevice('/dev/input/event4')

        # target = ['Forward', 'Right', 'Left']
        # labels ={'Forward':0, 'Right':1, 'Left':2}

        # Initialize Donkey car,..
        kit.continuous_servo[0].throttle = 0
        kit.servo[1].angle = 107

        lane_model_path = ['/home/ponata/A1-PONATA/Hayoung/lane_model_test/lane_model_v3-03.yaml',
                           "/home/ponata/A1-PONATA/Hayoung/lane_model_test/lane_model_v3-03.h5"]
        lane_model = load_model(lane_model_path)

        # Load motion model...
        motion_model_path = ['/home/ponata/A1-PONATA/Hayoung/motion_model_test/mm_v6.yaml',
                             "/home/ponata/A1-PONATA/Hayoung/motion_model_test/mm_v6.h5"]
        motion_model = load_model(motion_model_path)

        print("Initial Settings are done.\n")

        # START THREAD
        while True:
            main()