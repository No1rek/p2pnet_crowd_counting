import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset as TDataset, DataLoader
from tqdm import tqdm
from scipy.io import loadmat


@torch.no_grad()
class DataTransformer:
    def __init__(self, ):
        self.norm_means = [0.485, 0.456, 0.406]
        self.norm_stds = [0.229, 0.224, 0.225]
        
        self.scale_lower = 0.7
        self.scale_upper = 1.3
        self.min_size = 128
        
        self.crop_size = 128
    
    def cut_to_multiple_of_16(self, image, targets):
        """
            Cuts image so it's side's legths are divisible by 16
        """
        C,H,W = image.shape
        H1 = H // 16 * 16
        W1 = W // 16 * 16
        image = image[:, :H1, :W1]
        h = targets[:, 0]
        w = targets[:, 1]
        targets = targets[(h < H1) & (w < W1)]
        return image, targets
        
    def normalize(self, img, targets):
        img_t = T.functional.normalize(img, mean=self.norm_means,
                                        std=self.norm_stds)
        return img_t, targets
    
    def scale(self, img, targets):
        """
            Applies random scaling to image and labels
        """
        C,H,W = img.shape
        scale = np.random.uniform(self.scale_lower, self.scale_upper)
        if min(W,H) < self.min_size or min(H*scale, W*scale) < self.min_size:
            img_t = T.functional.resize(img, self.min_size)
        else:
            img_t = T.functional.resize(img, (int(H*scale), int(W*scale)))
        targets_t = targets*scale
        return img_t, targets_t
    
    def crop(self, img, targets):
        """
            Performs random cropping
        """
        C,H,W = img.shape
        if H > self.crop_size: top = np.random.randint(0, H-self.crop_size)
        else: top = 0
        if W > self.crop_size: left = np.random.randint(0, W-self.crop_size)
        else: left = 0

        img_t = T.functional.crop(img, top, left, self.crop_size, self.crop_size)

        # filter points to leave only ones in cropped img
        h = targets[:,0]
        w = targets[:,1]
        mask = (w >= left) & (w <= left+self.crop_size) & (h >= top) & (h <= top+self.crop_size)

        targets_t = np.copy(targets[mask, :])
        # shift coordinates to bound them to cropped img size
        targets_t[:, 0] -= top
        targets_t[:, 1] -= left 
        return img_t, targets_t
    
    def flip(self, img, targets):
        """
            Implements random flipping with probability of 0.5
        """
        if bool(np.random.randint(0, 2)):
            C,H,W = img.shape
            img_t = T.functional.hflip(img)
            targets_t = np.copy(targets)
            targets_t[:, 1] = W - targets_t[:, 1]
            return img_t, targets_t
        return img, targets
    
class Dataset(TDataset):
    def __init__(self, paths, transformer, crop=False, keep_src=False):
        """
            Params:
            transformer - object that performs image transformations
            crop - if True: performs cropping and scaling  when get data 
            keep_src - whether to store original images
        """
        super().__init__()
        self.crop = crop
        self.transformer = transformer
        self.keep_src = keep_src
        
        self.image_paths = []
        self.label_paths = []
        self.images = []
        self.images_src = []
        self.labels = []
        self.size = len(paths)
        
        for ip, lp in tqdm(paths):
            image_src, label_src = self.load_image_label(ip,lp)
            if keep_src:
                self.images_src.append(image_src)
            image, label = self.transformer.normalize(image_src, label_src)
            self.images.append(image)
            self.labels.append(label)                
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.crop:
            image, label = self.transformer.scale(image, label)
            image, label = self.transformer.crop(image, label)
            image, label = self.transformer.flip(image, label)
        else:
            image, label = self.transformer.cut_to_multiple_of_16(image, label)
        return image, torch.as_tensor(label, dtype=torch.float)

    def view(self, idx):
        assert self.keep_src, "Dataset doesn't contain source images"
        image = self.images_src[idx]
        return image
    
    def load_image_label(self, img_path, label_path):
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = T.functional.to_tensor(img).detach()
        label = loadmat(label_path)['image_info'][0][0][0][0][0]
        label = label[:, [1,0]] # w,h to h,w to match pil format
        return img, label