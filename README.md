# FaceRecognition
Ubuntu 18.04
### 1. Create Anaconda environment (optional)
    conda create -n Camera_env python=3.6.12
### 2. install python libs
    pip install numpy==1.19.2
    pip install matplotlib==3.3.4
    pip install opencv-contrib-python==4.5.2
    pip install tensorflow==2.3.0
    pip install face_recognition
    
### 3. Install cuda toolkit and cudnn for nvidia gpu support (cái này em ko nhớ rõ cách cài đặt lắm, anh thử theo hướng dẫn này xem ok ko ạ)
    * install cuda toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda
    * install cudnn
    In order to download cuDNN libraries, you need to go to https://developer.nvidia.com/cudnn and click on the Download cuDNN button. The webpage will ask you to login into the NVIDIA developer account. After logging in and accepting their terms and conditions, Download 10.1 runtime & developer library for 18.04 (Files cuDNN7.6.5 Runtime Library for Ubuntu18.04 (Deb) & cuDNN v7.6.5 Developer Library for Ubuntu18.04 (Deb)). Then:
    sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
    sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
    
    * Check installation
    1. Open termial, run:
        nvidia-smi 
        You will see "CUDA Version: 10.1" in the top-right conner
    2. Open python, run
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        
        You can see name: "/device:GPU:0" in devices list
        
### 4. Install jupyter and run Demo.ipynb notebook
