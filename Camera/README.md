# CSI-Camera
Simple example of using a MIPI-CSI(2) Camera (like the Raspberry Pi Version 2 camera) with the NVIDIA Jetson Nano Developer Kit.
The camera should be installed in the MIPI-CSI Camera Connector on the carrier board. The pins on the camera ribbon should face the Jetson Nano module.

To test the camera:

```
$ gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e

```

There are three examples:

simple_camera.py is a Python script which reads from the camera and displays to a window on the screen using OpenCV:

$ python simple_camera.py

face_detect.py is a python script which reads from the camera and uses  Haar Cascades to detect faces and eyes:

$ python face_detect.py

Haar Cascades is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. The function is then used to detect objects in other images. 

See: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html 

The third example is a simple C++ program which reads from the camera and displays to a window on the screen using OpenCV:

```
$ g++ -std=c++11 -Wall -I/usr/lib/opencv simple_camera.cpp -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_videoio -o simple_camera

$ ./simple_camera
```


<h2>Notes</h2>

<h3>Camera Image Formats</h3>
You can use v4l2-ctl to determine the camera capabilities. v4l2-ctl is in the v4l-utils:

$ sudo apt-get install v4l-utils

For the Raspberry Pi V2 camera the output is (assuming the camera is /dev/video0):

```
$ v4l2-ctl --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Index       : 0
	Type        : Video Capture
	Pixel Format: 'RG10'
	Name        : 10-bit Bayer RGRG/GBGB
		Size: Discrete 3280x2464
			Interval: Discrete 0.048s (21.000 fps)
		Size: Discrete 3280x1848
			Interval: Discrete 0.036s (28.000 fps)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.017s (60.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.017s (60.000 fps)
```

<h3>GStreamer Parameter</h3>
For the GStreamer pipeline, the nvvidconv flip-method parameter can rotate/flip the image. This is useful when the mounting of the camera is of a different orientation than the default.

```

flip-method         : video flip methods
                        flags: readable, writable, controllable
                        Enum "GstNvVideoFlipMethod" Default: 0, "none"
                           (0): none             - Identity (no rotation)
                           (1): counterclockwise - Rotate counter-clockwise 90 degrees
                           (2): rotate-180       - Rotate 180 degrees
                           (3): clockwise        - Rotate clockwise 90 degrees
                           (4): horizontal-flip  - Flip horizontally
                           (5): upper-right-diagonal - Flip across upper right/lower left diagonal
                           (6): vertical-flip    - Flip vertically
                           (7): upper-left-diagonal - Flip across upper left/low
```

<h2>OpenCV and Python</h2>
Starting with L4T 32.2.1 / JetPack 4.2.2, GStreamer support is built in to OpenCV.
The OpenCV version is 3.3.1 for those versions. Please note that if you are using
earlier versions of OpenCV (most likely installed from the Ubuntu repository), you
will get 'Unable to open camera' errors.
<br>
If you can open the camera in GStreamer from the command line, and have issues opening the camera in Python, check the OpenCV version. 

```
>>>cv2.__version__
```

<h2>Release Notes</h2>

v2.0 Release September, 2019
* L4T 32.2.1 (JetPack 4.2.2)
* OpenCV 3.3.1
* Tested on Jetson Nano

Initial Release (v1.0) March, 2019
* L4T 32.1.0 (JetPack 4.2)
* Tested on Jetson Nano


