# OCLReID
This project is for target person tracking based on [mmtrack](https://github.com/open-mmlab/mmtracking) framework. For running this code with robot/rosbag, please refer to [OCL-RPF](https://github.com/MedlarTea/OCL-RPF)

## Install

### For Video Running Only

Create a conda environment and install OCLReID (based on mmtrack), worked in RTX3090
```bash
git clone https://github.com/MedlarTea/OCLReID
cd OCLReID
conda create -n oclreid python=3.7
conda activate oclreid
conda install pytorch=1.11 cudatoolkit=11.3 torchvision=0.12.0 -c pytorch
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.26.0
pip install -r requirements.txt
pip install -r requirements/build.txt
pip install -v -e .

# install orientation estimation method
cd mmtrack/models/orientation
pip install -r requirements.txt
pip install -v -e .
```

Download pre-trained weights for OCLReID
  - Download 2d joint detection models: [Google drive](https://drive.google.com/drive/folders/1v-2Noym5U13BG6Zwj9EoqYRn6GXimh6p?usp=sharing) and put the checkpoints to `OCLReID/mmtrack/models/pose/Models/sppe`.
  - Download ReID models: [Google drive](https://drive.google.com/file/d/1cjqnHFcYzFZvzLrqvzry6Bgt8mWaWILg/view?usp=drive_link), then make directory `OCLReID/checkpoints/reid` and put the checkpoints to it.


## Run It!

### Video Running
```bash
cd OCLReID
python run_video.py --show_result
```
This would run the `./demo.mp4`.

### Run on the customized dataset
Our customized dataset is provided in `dataset` directory with four scenarios: `corridor1`, `corridor2`, `lab_corridor` and `room`. We provide `raw_video.mp4` and `labels.txt` for each scenario. Specifically, bbox annotations in the `label.txt` are represented as `x1,y1,w,h`.

*Note: the annotations are rough, but should be enough for evaluating the ReID performance of algorithms.*


## Citation
```
@article{ye2024oclrpf,
  title={Person re-identification for robot person following with online continual learning},
  author={Ye, Hanjing and Zhao, Jieting and Zhan, Yu and Chen, Weinan and He, Li and Zhang, Hong},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```