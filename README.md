

# Quest3-Teleoperation for Franka Robot

The code is for data collection framework using Meta Quest 3.

Please refer to https://github.com/XR-Robotics and https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python for teleoperation with controller tracking.


## Setup Instruction


### Install the XR App on Headset

Step-1: Turn on developer mode on Meta Quest3 headset, and make sure that adb is installed on PC properly.

Step-2: Download on a PC with adb installed.

Step-3: To install apk on the headset, use command


Please refer to https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest.


```bash
 adb install -g XRoboToolkit-PICO-1.1.1.apk
```

### Install XRoboToolkit-PC-Service

Step-1: Download XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb

Step-2: To install, use command

```bash
sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb
```

Please refer to https://github.com/XR-Robotics/XRoboToolkit-PC-Service.


#### VR Robot Client Setup:
```bash
mkdir vr_robot_client/build && cd vr_robot_client/build
cmake -DFRANKA_INSTALL_PATH=/your/path/to/libfranka/install ..
make -j4
```


## Usage Instruction

