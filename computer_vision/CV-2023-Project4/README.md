Computer Vision - Project 4 - Darts Detection
===

## Dependencies

The required **Python** version is **3.11**.
The dependencies required to run this application are specified in `requirements.txt`,
which has the following contents:
```
certifi==2023.7.22
charset-normalizer==3.2.0
cmake==3.27.2
contourpy==1.1.0
cycler==0.11.0
dill==0.3.7
filelock==3.12.3
fonttools==4.42.1
idna==3.4
imageio==2.31.3
Jinja2==3.1.2
kiwisolver==1.4.5
lazy_loader==0.3
lit==16.0.6
MarkupSafe==2.1.3
matplotlib==3.7.1
mpmath==1.3.0
networkx==3.1
numpy==1.24.2
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.2.10.91
nvidia-cusolver-cu11==11.4.0.1
nvidia-cusparse-cu11==11.7.4.91
nvidia-nccl-cu11==2.14.3
nvidia-nvtx-cu11==11.7.91
opencv-python==4.8.0.76
packaging==23.1
pandas==2.1.0
Pillow==10.0.0
psutil==5.9.5
py-cpuinfo==9.0.0
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3
PyWavelets==1.4.1
PyYAML==6.0.1
requests==2.31.0
scikit-image==0.21.0
scipy==1.11.2
seaborn==0.12.2
six==1.16.0
sympy==1.12
tifffile==2023.8.30
torch==2.0.1
torchvision==0.15.2
tqdm==4.66.1
triton==2.0.0
typing_extensions==4.7.1
tzdata==2023.3
ultralytics==8.0.168
urllib3==2.0.4
```

## How to run it?

Example:
```
python src/main.py --input ./data/fake_test_inputs/ --output ./data/fake_test_results/
```

The `fake_test_inputs` folder is expected to exist and to contain the task folders.
The `fake_test_results` folder is expected to exist. The directory structure will be created on the fly if it does
not exist.