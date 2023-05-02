Computer Vision - Project 1 - Double Dominoes
===

## Dependencies

The required **Python** version is **3.10**.
The dependencies required to run this application are specified in `requirements.txt`,
which has the following contents:
```
numpy==1.24.2
opencv-python==4.5.5.62
matplotlib==3.7.1
```

## How to run it?

Example (it will take about 3 minutes to run for 100 images):
```
python src/main.py --input ./data/fake_test_inputs/ --output ./data/fake_test_results/
```

The `fake_test_inputs` folder is expected to exist and to contain two sub-folders: `regular_tasks` and `bonus_task`.
The `fake_test_results` folder is expected to exist. The directory structure will be created on the fly if it does
not exist.