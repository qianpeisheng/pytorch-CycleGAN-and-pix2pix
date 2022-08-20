"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
# from PIL import Image
import SimpleITK as sitk
import cv2
import numpy as np
import random
import torch
from options.train_options import TrainOptions
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


import re


class BRATS18Dataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # data_root '/data1/ziyuan/huaizhe/BRATS' images are sliced to 128 slices to avoid black backgrounds
        self.A_dir = os.path.join(opt.dataroot, 'T1CE_NPY') # t1ce
        self.B_dir = os.path.join(opt.dataroot, 'T2_NPY') # t2
        self.A_paths = [f for f in os.listdir(self.A_dir) if 'label' not in f]
        self.A_label_paths = [re.sub('.npy','_label.npy',f) for f in self.A_paths]
        self.B_paths = [f for f in os.listdir(self.B_dir) if 'label' not in f]
        self.B_label_paths = [re.sub('.npy','_label.npy',f) for f in self.B_paths]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        # print('sizeA',self.A_size)
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform = transforms.Compose([transforms.ToPILImage(), # np array -> pil
                                            transforms.RandomRotation(15),
                                            transforms.ToTensor()])

        # self.transform = transforms.Compose([transforms.RandomRotation(15),
        #                                      transforms.Resize(280),
        #                                      transforms.RandomCrop(256),
        #                                      #transforms.RandomAffine(degree=10, translate=(0, 0.1), shear=(-45, 45),
        #                                      #                        scale=(0.8, 1.2)),
        #                                      transforms.ToTensor(),])
        """
        self.resize = None
        self.transform = None
        if opt.phase == 'train':
            if opt.preprocess != 'None':
                self.transform = iaa.Sequential([iaa.Resize({'width':1024, 'height': 'keep-aspect-ratio'}), iaa.CropToFixedSize(400,400)])
            else:
                self.resize = iaa.Resize({'width':400, 'height': 'keep-aspect-ratio'})
        else:
            self.resize = iaa.Resize({'width':400, 'height': 'keep-aspect-ratio'})
        """
        print('len mr path: {}, len ct path {}'.format(len(self.A_paths), len(self.B_paths)))
        # print('data size:', len(self.A_paths) )

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        """
        ce.shape = (120, 512, 511) t2.shape = (80, 448, 448) or (40, 348, 348)
        """
        # print(index % self.A_size)
        A_path = self.A_paths[index % self.A_size]
        A = np.load(os.path.join(self.A_dir, A_path))
        augmented_A = self.transform(np.copy(A))

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B= np.load(os.path.join(self.B_dir, B_path))
        augmented_B = self.transform(np.copy(B))

        """
        if self.transform:
            A = np.array(self.transform(images=A))
            B = np.array(self.transform(images=B))
        if self.resize:
            # resize both image and label

            A_label_path = self.A_label_paths[index % self.A_size]
            A_label = np.load(os.path.join(self.A_dir, A_label_path))
            B_label_path = self.B_label_paths[index_B]
            B_label = np.load(os.path.join(self.B_dir, B_label_path))

            segmapA = SegmentationMapsOnImage(A_label, shape=A.shape)
            segmapB = SegmentationMapsOnImage(B_label, shape=B.shape)

            A, A_label_resize = self.resize(images=A, segmentation_maps = segmapA)
            B, B_label_resize = self.resize(images=B, segmentation_maps = segmapB)

            # np.save(os.path.join(self.A_dir,re.sub('.npy','_400.npy', A_path)), A)
            # np.save(os.path.join(self.B_dir,re.sub('.npy','_400.npy', B_path)), B)

            np.save(os.path.join(self.A_dir,re.sub('.npy','_400.npy', A_label_path)), A_label_resize.arr[...,0])
            # np.save(os.path.join(self.B_dir,re.sub('.npy','_400.npy', B_label_path)), B_label_resize.arr[...,0])
            # print(B_label_resize.arr[...,0].shape,A.shape)
            print(A_path, A_label_path,re.sub('.npy','_400.npy', A_label_path))
        """
        # print('pathA: %s, pathB: %s'%(A_path,B_path))

        ret = {'A':  torch.tensor(A[np.newaxis, :]).float(), 'B': torch.tensor(B[np.newaxis, :]).float(),
            'augmented_A':  augmented_A.float(), 'augmented_B': augmented_B.float(),
            'A_paths': A_path, 'B_paths': B_path}
        return ret

    def __len__(self):
        """Return the total number of images."""
        # return self.A_size
        return 9600 # for testing purpose

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = BRATS18Dataset(opt)
    brats_dataloader = DataLoader(dataset, batch_size=opt.batch_size)
    for i, data in enumerate(brats_dataloader):
        print(i)

    # len dataset 9600
    #
    import pdb; pdb.set_trace()
    # data9 = Dataset[9]
    # print(data9['data_A'].shape, data9['data_B'].shape)
    print(Dataset.A_paths)
