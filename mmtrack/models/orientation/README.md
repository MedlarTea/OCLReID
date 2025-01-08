
# Quick start
## Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${HBOE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download checkpoints from ([OneDrive](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/czw390_psu_edu/EoXLPTeNqHlCg7DgVvmRrDgB_DpkEupEUrrGATpUdvF6oQ?e=CQQ2KY))
   ```
   mkdir checkpoints
   ```
   Put 'model_hboe.pth' under folder checkpoints.

## Using MEBOW to estimate orientation
   Change images folder and annotations file at mebow_estimate.sh.
   ```
   ./mebow_estimate.sh
   ```

## Using keypoints to estimate orientation
   Change images folder and annotations file at mebow_estimate.sh.
   ```
   ./keypoints_estimate.sh
   ```

## Acknowledgement
This repo is based on [MEBOW](https://chenyanwu.github.io/MEBOW/).