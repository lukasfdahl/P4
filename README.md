### About This Project
This Github covers our P4 project at AAU.
This project focuses on training models on compressed h.264 video data like residuals and motion vectors, as well as extracting those for training purposes.

### Structure
- (Slurm script for starting training on a server): run_train.slurm
- (Training script): train.py
- (Configs for the various training runs): experiment_yamls
- (Model): model.py
- (Dataloader): dataloader.py
- (Functions for evaluating the model and getting performance metrics): eval_framwork.py
- (Various helper and utility functions): helpers.py train_helpers.py data_helpers.py


### Download
The video download and motion vector/residual extraction are handled via a docker container. 
To run it download the container and use the following instructions (while updating the mappings to match the target device).

Link to the container: https://drive.google.com/file/d/1iAlU4W43IA4IKKCuTFJmZJd4GHucmStb/view?usp=sharing

Run instructions:
``` bash
docker run -it \
  -v /mnt/hdd:/mnt/hdd \
  -v /home/students/Desktop/projects/P4:/app \
  -w /app \
  localhost/video_env:latest \
  python download.py
```
