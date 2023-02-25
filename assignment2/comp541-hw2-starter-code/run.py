from CnnModel import SimpleCNN
from train import Trainer
from DataLoader import LoaderClass
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
from torchvision import transforms, utils
import numpy as np
import os
from PIL import Image, ImageFile

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


def load_artist_data():
    data = []
    labels = []
    artists = [x for x in os.listdir(DATA_PATH) if x != '.DS_Store']
    for folder in os.listdir(DATA_PATH):
        class_index = artists.index(folder)
        for image_name in os.listdir(DATA_PATH + "/" + folder):
            img = Image.open(DATA_PATH + "/" + folder + "/" + image_name)
            artist_label = (np.arange(NUM_ARTISTS) == class_index).astype(np.float32)
            data.append(np.array(img))
            labels.append(artist_label)
    shuffler = np.random.permutation(len(labels))
    data = np.array(data)[shuffler]
    labels = np.array(labels)[shuffler]

    length = len(data)
    val_size = int(length*0.2)
    val_data = data[0:val_size+1]
    train_data = data[val_size+1::]
    val_labels = labels[0:val_size+1]
    train_labels = labels[val_size+1::]
    data_dict = {"train_data":train_data, "val_data":val_data}
    label_dict = {"train_labels":np.array(train_labels), "val_labels":np.array(val_labels)}

    return data_dict, label_dict

if __name__ == "__main__":
    data, labels = load_artist_data()
    model = SimpleCNN (use_cuda=USE_CUDA, pooling=POOLING, use_dropout=use_dropout, dropout_prob=dropout_prob)
    if use_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    if use_SGD:
        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay)
    transforms = {
        'train': transforms.Compose([
            transforms.Resize(50),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(50),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # The following are added by me regarding the assignment section 1.7:
            transforms.RandomAffine(degrees=0, scale=(0.5, 0.75))
        ]),
        }
    train_dataset = LoaderClass(data, labels, "train", transforms["train"])
    valid_dataset = LoaderClass(data, labels, "val", transforms["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4, pin_memory=True)

    criterion = CrossEntropyLoss()

    if lr_scheduler == "stepLr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)
    elif lr_scheduler == "cosineAnnealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1, eta_min=10, last_epoch=- 1, verbose=False)

    trainer_m = Trainer(model, criterion, train_loader, val_loader, optimizer, num_epoch=NUM_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr_scheduler= lr_scheduler)
    best_model = trainer_m.train()
