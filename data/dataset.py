import os
import random
from random import choice

import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from utils.read_flow import readPFM


class SceneFlowDataset(Dataset):
    """
    params：
    image_dir: directory to save PNG images
    flow_dir: directory to save flow files(in PFM data format)
    random_seed: set random seed to select initial image and corresponding flows(each time we initialize the dataset,
                we can get probably the same random dataset)
    flow_prefix, flow_suffix:  different type of scene-flow optical-flow data have different prefix and suffix
    seq_len: sequence length
    train_vae: indicate what type of return values should be (when training 3DcVAE, image should concatenate
                with every flow in channel dimension, but training flow2rgb do not need that operation)
    """
    def __init__(self, image_dir, flow_dir, random_seed, flow_prefix='',
                 flow_suffix='', seq_len=16, train_vae=False, image_size=128, **kwargs):
        super(SceneFlowDataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.flow_dir = flow_dir
        self.seed = random_seed
        self.prefix = flow_prefix
        self.suffix = flow_suffix
        self.seq_len = seq_len
        self.train_vae = train_vae
        self.image_size = image_size

        self.images = os.listdir(image_dir)
        # self.flows = os.listdir(flow_dir)
        # 随机选择起始图像帧
        self.ids = set()
        nums = len(self.images)
        random.seed(random_seed)
        self.ids = random.sample(range(nums - seq_len), nums // 4)
        self.ids = [str(i).zfill(4)+'.png' for i in self.ids]
        # while len(self.ids) < nums // 4:  #
        #     id_ = choice(self.images)
        #     num = int(id_.split('.')[0])
        #     if num <= nums - seq_len:
        #         self.ids.add(id_)
        # self.ids = list(self.ids)  # 转为list

    def __len__(self):
        return len(self.ids)

    def image_preprocess(self, image):
        # image : PIL Image format
        # w, h = image.size
        # new_w, new_h = int(w * scale), int(h * scale)
        new_w, new_h = self.image_size, self.image_size
        image = image.resize((new_w, new_h), resample=PIL.Image.BICUBIC)
        image = np.asarray(image) / 255.  # H W C

        return image

    def flow_preprocess(self, flow):
        # flow H W C=3(channel 3 is all zero)
        # resize
        new_w, new_h = self.image_size, self.image_size
        c1 = flow[:, :, 0]
        c2 = flow[:, :, 1]
        image_c1 = Image.fromarray(c1)  # 转为Image便于resize
        image_c1 = image_c1.resize((new_w, new_h), resample=PIL.Image.BICUBIC)  # resize
        image_c1 = np.expand_dims(np.asarray(image_c1), axis=2)  # 转回ndarray并扩展通道维
        image_c2 = Image.fromarray(c2)
        image_c2 = image_c2.resize((new_w, new_h), resample=PIL.Image.BICUBIC)
        image_c2 = np.expand_dims(np.asarray(image_c2), axis=2)
        resize_flow = np.concatenate([image_c1, image_c2], axis=2)
        # normalize
        max_, min_ = resize_flow.max(), resize_flow.min()
        resize_flow = (resize_flow - min_) / (max_ - min_)  # normalize to 0-1

        return resize_flow

    def __getitem__(self, item):
        name = self.ids[item]  # image x0 filename
        idx = int(name.split('.')[0])  # image x0

        images, flows = [], []

        for i in range(idx, idx + self.seq_len):
            temp_idx = str(i).zfill(4)  # 构造4位数字的字符串，如0001,0002,...zfill补全字符串数字前面的0
            img_path = os.path.join(self.image_dir,  temp_idx + '.png')
            img = Image.open(img_path)  # H, W, C=3
            img = self.image_preprocess(img)
            images.append(img)

            flow_path = os.path.join(self.flow_dir, self.prefix + temp_idx + self.suffix)
            # 例如，train_flow2rgb_test中，self.prefix='OpticalFlowIntoFuture_', self.suffix='_L.pfm'
            flow, _ = readPFM(flow_path)
            flow = self.flow_preprocess(flow)
            flows.append(flow)  # 第3个通道全为0, flow:H,W,C=2

        # 最后一张图
        img_path = os.path.join(self.image_dir, str(idx + self.seq_len).zfill(4) + '.png')
        img = Image.open(img_path)  # H, W, C=3
        img = self.image_preprocess(img)
        images.append(img)

        if self.train_vae:
            flows_add = []
            # 训练VAE时，需将初始图像帧x0(images[0]), concat到每一个flow中（通道维）
            for i in range(self.seq_len):
                flows_add.append(np.concatenate([images[0], flows[i]], axis=2))

            return {
                'image0': torch.as_tensor(images[0].copy()).permute(2, 0, 1).contiguous().float(),  # C H W
                'flows': torch.as_tensor(flows.copy()).permute(3, 0, 1, 2).contiguous().float(),  # C,T=seq_len,H W
                'flows_add': torch.as_tensor(flows_add.copy()).permute(3, 0, 1, 2).contiguous().float()
            }

        return {
            'images': torch.as_tensor(images.copy()).permute(0, 3, 1, 2).contiguous().float(),  # T=seq_len+1, C H W
            'flows': torch.as_tensor(flows.copy()).permute(0, 3, 1, 2).contiguous().float()  # T=seq_len, C H W
        }
        # both return values are normalize to 0-1
