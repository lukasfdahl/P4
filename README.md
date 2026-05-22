### About This Project
This Github covers our P4 project at AAU.
This project focuses on training models on compressed h.264 video data like residuals and motion vectors, as well as extracting those for training purposes.

### Structure
- (Script for starting training on the server): run_train.slurm
- (Training script): train.py
- (Configs for the various training runs): experiment_yamls
- (Model): model.py
- (Dataloader): dataloader.py
- (Functions for evaluating the model and getting performance metrics): eval_framwork.py
- (Various helper and utility functions): helpers.py train_helpers.py
