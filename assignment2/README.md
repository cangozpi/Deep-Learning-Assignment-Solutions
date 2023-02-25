# Assignment 2: Convolutional Neural Networks

This repository contains my solutions to _Assignment 2: Convolutional Neural Networks_ of COMP 541 (Deep Learnign) course offered in Koc University.

Check out _comp541-hw2.pdf_ for instructions on the homework to understand the implementations. _comp541_assignment2_cgozpinar18.pdf_ contains my answers to the Latex parts of the homework.

---

## Section 1

_comp541-hw2-starter-code_ folder contains the code for the solutions of the _Section 1_ of the questions in the homework instructions.

Code can be run by:

```bash
python3 run.py
```

**Note**: Before you can run the code, you need to download the dataset for this section as instructed by the \_comp541-hw2.pdf\_\_.

Experimentations explained in the documents can be reproduced by playing with the hyperparameters available in the _run.py_ as follows:

```python
# Parameters ---------------
LR = 1e-3
Momentum = 0.9 # If you use SGD with momentum
BATCH_SIZE = 128
USE_CUDA = False
POOLING = True
NUM_EPOCHS = 200
PATIENCE = 50
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.2
NUM_ARTISTS = 11
DATA_PATH = "./art_data/artists"
ImageFile.LOAD_TRUNCATED_IMAGES = True # Do not change this
# Dropout params
use_dropout = True
dropout_prob = 0.25
# Optimizer choice
use_ADAM = True
use_SGD = False
weight_decay = 0
# lr scheduling
lr_scheduler = None # possible options are ['stepLr', 'cosineAnnealing', None]
# --------------------------
```

---

## Section 2

_part2-code_ folder contains the code for the solutions of the _Section 1_ of the questions in the homework instructions.

1. ### Downloading Dataset

   ***

   Goal is to download face images from the given urls in the './part2-data/subset_actors.txt', './part2-data/subset_actresses.txt' files. The faces inside the corresponding URL's are cropped and then saved with the name of format '{Celebrity_name}-{unique_id}.png' into 'part2-data' folder.

   **Parameters:**

   - _timeout_: number of seconds to wait before terminating image download request.
   - _saved_img_folder_path_: directory to save the downloaded images.
   - _max_thread_workers_: number of workers to use for threading. This is used for speeding up downloading images using threads.

   **To run:**

   ```bash
   python download_data_script.py
   ```

   ***

2. ### Running the code

   Code for this section can be run by running _main.py_.

   ```bash
   python main.py
   ```

   Which section of the homework you run can be controlled by setting the hyperparameters in the _main.py_ file as follows:

   ```python
   # Hyperparameters ---------------
   epochs = 20
   lr = 1e-2
   batch_size = 64
   num_workers = 4
   assignment_section = 2.5 # choose from [2.1, 2.2, 2.3, 2.4, 2.5]
   # ----------------------
   ```

   _assignment_section_ parameter controls solution to which section of the homework instructions will be run. For example, setting _assignment_section_ to _2.1_ runs the solution to _section 2.1_, whereas setting it to _2.5_ and running the _main.py_ file would result in the solution for _section 2.5_ to be run.
