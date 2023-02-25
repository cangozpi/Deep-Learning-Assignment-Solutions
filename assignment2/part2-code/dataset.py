import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import os


# Parameters: -----------
dataset_path = './part2-faceDataset'
train_data_num = 70
val__to_test_dataset_ratio = 0.4


class FaceDataset(Dataset):
    def __init__(self, dataset_dict, unique_names, dataset_path, preprocessing_transforms):
        """
        dataset_dict = {} # key (str):celeb_name, value(list): names of image names used for training/validating/testing
        unique_names (list of str): contains names of celebrities in the dataset
        """
        super().__init__()
        self.dataset_dict = dataset_dict 
        self.unique_names = unique_names

        self.one_hot_celeb_name = {} # one hot encoding of celeb names. key (str): celebrity name, value (torch.tensor): one hot encoding of the name
        for celeb_name in unique_names:
            self.one_hot_celeb_name[celeb_name] = torch.tensor([celeb_name == name for name in self.unique_names], dtype=torch.int8)

         # Preprocessing for AlexNet
        self.preprocessing_transforms = preprocessing_transforms

        self.dataset_size = celeb_face_nums = sum([len(v) for v in self.dataset_dict.values()])
        self.images = self.get_images()

       
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.images[idx] # (image as torch.Tensor, one_hot encoded class as torch.Tensor)

    def get_images(self):
        images = []
        for celeb_name in self.unique_names:
            celeb_file_names = self.dataset_dict[celeb_name]
            for file_name in celeb_file_names:
                img = torchvision.io.read_image(os.path.join(dataset_path, file_name))
                img = transforms.ToPILImage()(img).convert('RGB')
                img = self.preprocessing_transforms(img)
                images.append((img, self.one_hot_celeb_name[celeb_name]))

        return images



def get_train_val_test_fileNames(dataset_path, train_data_num, val__to_test_dataset_ratio):
    # check if dataset folder exists
    if os.path.exists(dataset_path):
        # get names of the images
        image_file_names = os.listdir(dataset_path)
        # extract unique celebrity names
        unique_names = set(map(lambda x: x.split('-')[0], image_file_names))

        celeb_face_dict = {} # key (str):celebrity name, value(list): names of image names for the correspondnig celebrity name
        for name in unique_names:
            celeb_face_dict[name] = [] # initialize as list
    
        for file_name in image_file_names:
            cur_celeb_name = file_name.split('-')[0]
            celeb_face_dict[cur_celeb_name].append(file_name)


        # Divide into separate datasets ---
        train_dataset_dict = {} # key (str):celeb_name, value(list): names of image names used for training 
        val_dataset_dict = {} # key (str):celeb_name, value(list): names of image names used for validating
        test_dataset_dict = {} # key (str):celeb_name, value(list): names of image names used for testing
        
        # initialize with empty dicts
        for name in unique_names:
            train_dataset_dict[name] = []
            val_dataset_dict[name] = []
            test_dataset_dict[name] = []

        # fill the dictionaries
        for celeb_name in celeb_face_dict:
            celeb_file_names = celeb_face_dict[celeb_name]
            num_celeb_images = len(celeb_file_names)
            
            cur_val_data_num = int((num_celeb_images - train_data_num) * val__to_test_dataset_ratio)
            cur_test_data_num = num_celeb_images - train_data_num - cur_val_data_num

            train_dataset_dict[celeb_name] = [celeb_file_names[i] for i in range(train_data_num)]
            val_dataset_dict[celeb_name] = [celeb_file_names[i] for i in range(train_data_num, train_data_num + cur_val_data_num)]
            test_dataset_dict[celeb_name] = [celeb_file_names[i] for i in range(train_data_num + cur_val_data_num, num_celeb_images)]

        return train_dataset_dict, val_dataset_dict, test_dataset_dict, unique_names
    else:
        raise Exception(f"Face dataset not found at path: {dataset_path}. Make sure you run download_data_script.py first.")
        return None



def get_datasets(preprocessing_transforms):
    # -----------------
    train_dataset_dict, val_dataset_dict, test_dataset_dict, unique_names = get_train_val_test_fileNames(dataset_path, train_data_num, val__to_test_dataset_ratio)
    # NOTE:
    # train_dataset_dict = {} # key (str):celeb_name, value(list): names of image names used for training 
    # val_dataset_dict = {} # key (str):celeb_name, value(list): names of image names used for validating
    # test_dataset_dict = {} # key (str):celeb_name, value(list): names of image names used for testing
    # ------------------

    train_dataset = FaceDataset(train_dataset_dict, unique_names, dataset_path, preprocessing_transforms)
    val_dataset = FaceDataset(val_dataset_dict, unique_names, dataset_path, preprocessing_transforms)
    test_dataset = FaceDataset(test_dataset_dict, unique_names, dataset_path, preprocessing_transforms)

    return train_dataset, val_dataset, test_dataset 


def get_dataLoaders(preprocessing_transforms, batch_size=32, num_workers=4):
    train_dataset, val_dataset, test_dataset = get_datasets(preprocessing_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, train_dataset.unique_names, train_dataset.one_hot_celeb_name




