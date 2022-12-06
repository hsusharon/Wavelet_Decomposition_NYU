# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from PIL import Image
from io import BytesIO
import random
import csv
import matplotlib.pyplot as plt



def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}

class RandomGamma(object):
    """
    Apply Random Gamma Correction to the images
    """
    def __init__(self, gamma=0):
        self.gamma = gamma

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if self.gamma == 0:
            return {'image': image, 'depth': depth}
        else:
            gamma_ratio = random.uniform(1 / self.gamma, self.gamma)
            return {'image': TF.adjust_gamma(image, gamma_ratio, gain=1),
                    'depth': depth}

from zipfile import ZipFile
def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    # for name in input_zip.namelist():
    #     print("File name:", name)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}  ## the image matrix not yet extracted
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))   
    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]
    print('Loaded training datasize of {0}.'.format(len(nyu2_train)))
    # tempkeys = data.keys()
    # print("size of data:", len(data), type(data.keys))
    return data, nyu2_train

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    # def __getitem__(self, idx):
    #     sample = self.nyu_dataset[idx]
    #     image = Image.open( BytesIO(self.data[sample[0]]) )
    #     depth = Image.open( BytesIO(self.data[sample[1]]) )
    #     sample = {'image': image, 'depth': depth}
    #     print("image size:", len(image), "depth size:", len(depth))
    #     if self.transform: sample = self.transform(sample)
    #     return sample

    def unzip_data(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        sample = {'image': image, 'depth': depth}
        # print("image size:", image, "depth size:",depth)
        print("transform:", self.transform)
        if self.transform: sample = self.transform(sample)

        plt.figure()
        plt.imshow(sample['image'])
        plt.show()
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self,is_test=False, is_224=False):
        self.is_test = is_test
        self.is_224 = is_224

    def __call__(self, sample):
        crop_size = 16
        image, depth = sample['image'], sample['depth']
        image = image.crop((crop_size, crop_size, 640-crop_size, 480-crop_size))

        if self.is_224:
            image = image.resize((224, 224))
        else:
            image = image.resize((640, 480))

        image = self.to_tensor(image)

        depth = depth.crop((crop_size, crop_size, 640-crop_size, 480-crop_size))
        # depth = depth.resize((512, 384))
        # image = image.resize((304, 224))
        if self.is_224:
            depth = depth.resize((224, 224))
        else:
            depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:            
            depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range [0.1m, 10m]
        depth = torch.clamp(depth, 10, 1000) # sets depth between 0.1m and 10m. [0, 1] -> [0, 1000] = [0m, 10m]

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

class NormalizeImage(object):
    """
    Apply Random Gamma Correction to the images
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample_tensors):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        image_tensor, depth_tensor = sample_tensors['image'], sample_tensors['depth']
        return {'image': TF.normalize(image_tensor, self.mean, self.std, self.inplace),
                'depth': depth_tensor}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def getNoTransform(is_test=False, is_224=False):
    transforms_list = [ToTensor(is_test=is_test, is_224=is_224)]
    # if normalize_input:
    #     transforms_list.append(NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transforms_list)

def getDefaultTrainTransform(is_224=False):
    transforms_list = [
        RandomHorizontalFlip(),
        RandomChannelSwap(0.1),
        RandomGamma(0.8),
        ToTensor(is_224=is_224)
    ]
    # if normalize_input:
    #     transforms_list.append(NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transforms_list)

def getTrainingTestingData(batch_size, num_workers=8, is_224=False):
    data, nyu2_train = loadZipToMem('nyu_data.zip')
    ## data is the loaded images not yet unzipped, nyu2_train is the list of filenames for training dataset
    print("Done loading zip file")

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform(is_224=is_224))
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform(is_224=is_224))
    training_data = []
    testing_data = []
    for i in range(30000): #2
        training_data.append(transformed_training.unzip_data(i))
        testing_data.append(transformed_testing.unzip_data(i+3000))
        if i%150 == 0:
            print("Processing until image:", i)
    # print("Training data size:", len(training_data), len(testing_data))
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(testing_data, batch_size, shuffle=False, num_workers=num_workers)
    
    
    return train_dataloader, test_dataloader

# reading data directly from file not with zip file
def getTrainingTestData_unzip(batch_size, num_workers=8, is_224 = False):
    print("Loading dataset ... ")
    ## load training dataset
    train_folder = "./data/data/nyu2_train.csv"
    training_list = []
    with open(train_folder) as file_obj:  ## read the train data list
        train_files = csv.reader(file_obj)
        for row in train_files:
            temp = []
            temp.append('./data/' + row[0])
            temp.append('./data/' + row[1])
            training_list.append(temp)
    training_dataset = []
    transform=getDefaultTrainTransform(is_224=is_224)
    for filename in training_list[0:2000]:  ## read the image for the training dataset
        image = Image.open(filename[0])
        depth = Image.open(filename[1])
        sample = {'image': image, 'depth': depth}  ## save the corres image and depth into a dict
        if transform: sample = transform(sample)
        training_dataset.append(sample)
    ## load testing dataset
    test_folder = "./data/data/nyu2_test.csv"
    testing_list = []
    with open(test_folder) as file_obj:  ## read the test data list
        train_files = csv.reader(file_obj)
        for row in train_files:
            temp = []
            temp.append('./data/' + row[0])
            temp.append('./data/' + row[1])
            testing_list.append(temp)
    testing_dataset = []
    transform=getNoTransform(is_224=is_224)
    for filename in testing_list[0:200]:  ## read the image for the training dataset
        image = Image.open(filename[0])
        depth = Image.open(filename[1])
        sample = {'image': image, 'depth': depth}  ## samve the cor image and depth into a dict
        if transform: sample = transform(sample)
        testing_dataset.append(sample)
    print("Loaded training_datasize:", len(training_dataset), "testing_datasize:", len(testing_dataset))
    train_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testing_dataset, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader