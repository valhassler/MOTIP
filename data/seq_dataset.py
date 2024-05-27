# Copyright (c) RuopengGao. All Rights Reserved.
# About:


import os
import cv2
import decord
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

# class VideoDataset(Dataset):
#     def __init__(self, video_path):
#         # Initialize the VideoReader
#         self.vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # Load video in CPU memory
#         self.length = len(self.vr)  # Total number of frames
    
#     def split_frame(self, np_array):
#         entire_image = np_array
#         top_view = entire_image[0:540, 62:892]           # Crop from top view
#         coming_in_view = entire_image[540:1500, 0:540]   # Crop from left side
#         going_out_view = entire_image[540:1500, 540:1080] # Crop from right side
#         return top_view, coming_in_view, going_out_view, entire_image

#     def __getitem__(self, idx):
#         if idx < 0 or idx >= self.length:
#             raise IndexError("Index out of bounds")
#         frame = self.vr[idx].asnumpy()
#         return self.split_frame(frame) #output: top_view, coming_in_view, going_out_view, entire_image

#     def __len__(self):
#         return self.length


class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, dataset: str, height: int = 800, width: int = 1333, view: str = 'non_specifc_view'):
        """
        Args:
            seq_dir:
            dataset: DanceTrack or MOT17 or et al.
        """
        print(f'Loading video from view: {view}')
        video_path = seq_dir
        self.vr = decord.VideoReader(video_path, ctx=decord.cpu(0)) 
        image_shape = self.vr[0].shape
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.length = len(self.vr)
        self.view = view
    
    def split_frame(self, entire_image):
        top_view = entire_image[0:540, 62:892]           # Crop from top view
        coming_in_view = entire_image[540:1500, 0:540]   # Crop from left side
        going_out_view = entire_image[540:1500, 540:1080] # Crop from right side
        return top_view, coming_in_view, going_out_view

    def process_image(self, image):
        ori_image = image.copy()
        h, w = image.shape[:2]
        scale = self.image_height / min(h, w)
        if max(h, w) * scale > self.image_width:
            scale = self.image_width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        # image = image.unsqueeze(0)
        return image, ori_image

    def __getitem__(self, idx):
        frame = self.vr[idx].asnumpy()
        if self.view in ['top', 'coming_in','going_out']: #when one wants to split here he has to work 
            splits = self.split_frame(frame)
            if self.view == 'top':
                image = splits[0]
            elif self.view == 'coming_in':
                image = splits[1]
            elif self.view == 'going_out':
                image = splits[2]
        else:
            image = frame
        print(f'Processing frame {idx}', end='\r')
        return self.process_image(image=image)

    def __len__(self):
        return self.length

# class SeqDataset(Dataset):
#     def __init__(self, seq_dir: str, dataset: str, height: int = 800, width: int = 1333):
#         """
#         Args:
#             seq_dir:
#             dataset: DanceTrack or MOT17 or et al.
#         """
#         image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
#         image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
#         self.image_paths = image_paths
#         self.image_height = height
#         self.image_width = width
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]
#         return

#     @staticmethod
#     def load(path):
#         """
#         Args:
#             path:

#         Returns:
#         """
#         # label_path = path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
#         image = cv2.imread(path)
#         assert image is not None
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return image

#     def process_image(self, image):
#         ori_image = image.copy()
#         h, w = image.shape[:2]
#         scale = self.image_height / min(h, w)
#         if max(h, w) * scale > self.image_width:
#             scale = self.image_width / max(h, w)
#         target_h = int(h * scale)
#         target_w = int(w * scale)
#         image = cv2.resize(image, (target_w, target_h))
#         image = F.normalize(F.to_tensor(image), self.mean, self.std)
#         # image = image.unsqueeze(0)
#         return image, ori_image

#     def __getitem__(self, item):
#         image = self.load(self.image_paths[item])
#         return self.process_image(image=image)

#     def __len__(self):
#         return len(self.image_paths)
