import os
import argparse
import torch
import torchvision.models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from data.dataset import SceneFlowDataset
from utils.read_flow import readPFM
from model import vgg_conv4_1 as vgg
from utils.evaluate import evaluate
from utils.vgg19 import get_features


def get_args():
    parser = argparse.ArgumentParser(description='Train Flow to RGB network')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained model path to load.')
    parser.add_argument('--image_dir', type=str, default='', help='Directory to read PNG images.')
    parser.add_argument('--flows_dir', type=str, default='', help='Directory to read optical-flow file.')
    parser.add_argument('--seq_len', type=int, default=16, help='Number of images to be predicted.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for select samples.')
    parser.add_argument('--flow_prefix', type=str, default='OpticalFlowIntoFuture_', help='Prefix of flow file.')
    parser.add_argument('--flow_suffix', type=str, default='_L.pfm', help='Suffix of flow file.')
    parser.add_argument('--train_vae', type=bool, default=False, help='Indicator of training VAE or not.')
    parser.add_argument('--val_percent', type=float, default=0.1, help='Percentage of validation set.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--beta', type=float, default=0.9, help='Momentum term for Adam.')
    parser.add_argument('--image_size', type=int, default=128, help='Image size.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='Directory to save models.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # test SceneFlowDataset
    dataset = SceneFlowDataset(image_dir='/media/tsy/I/scene flow/train_flow2rgb_test/images',
                               flow_dir='/media/tsy/I/scene flow/train_flow2rgb_test/flows',
                               random_seed=2,
                               flow_prefix='OpticalFlowIntoFuture_',
                               flow_suffix='_L.pfm',
                               seq_len=16,
                               train_vae=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # test training process
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    netC = vgg.Conv4_1(in_channels=3, channels=256)
    netF = vgg.Conv4_1(in_channels=2, channels=256)
    netD = vgg.InvertConcatConv4_1(channels=256, out_channels=3)

    netC.to(device=device)
    netF.to(device=device)
    netD.to(device=device)

    criterion = torch.nn.MSELoss()
    vgg19 = torchvision.models.vgg19(pretrained=True)
    vgg19.to(device=device)

    for batch in dataloader:
        seq_loss = 0
        images = batch['images']
        flows = batch['flows']
        images = images.to(device=device, dtype=torch.float32)
        flows = flows.to(device=device, dtype=torch.float32)
        image0 = images[:, 0, ...]
        # multiple steps for sequence
        for i in range(16):
            true_image = images[:, i + 1, ...]
            img_feat = netC(image0)
            flow_feat = netF(flows[:, i, ...])
            pred_image = netD(img_feat, flow_feat)
            # reconstruction loss
            loss = criterion(pred_image, true_image)
            # feature loss
            true_feats = get_features(vgg19, true_image)
            pred_feats = get_features(vgg19, pred_image)
            # loss = loss + criterion(pred_feats, true_feats)
            for j in range(5):
                loss += criterion(pred_feats[j], true_feats[j])

        val_loss = evaluate(netC, netF, netD, dataloader, device, args)

    # test dataset
    name = dataset.ids[0]  # image x0 filename
    idx = int(name.split('.')[0])  # image x0

    images, flows = [], []

    for i in range(idx, idx + dataset.seq_len):
        img_path = os.path.join(dataset.image_dir, str(i).zfill(4) + '.png')
        img = Image.open(img_path)  # H, W, C=3
        img = dataset.image_preprocess(img)
        images.append(img)

        flow_idx = str(i).zfill(4)  # 构造4位数字的字符串，如0001,0002,...zfill补全字符串数字前面的0
        flow_path = os.path.join(dataset.flow_dir, dataset.prefix + flow_idx + dataset.suffix)
        # 例如，train_flow2rgb_test中，self.prefix='OpticalFlowIntoFuture_', self.suffix='_L.pfm'
        flow, _ = readPFM(flow_path)
        flow = dataset.flow_preprocess(flow)
        flows.append(flow)  # 第3个通道全为0, flow:H,W,C=2

    # 最后一张图
    img_path = os.path.join(dataset.image_dir, str(idx + dataset.seq_len).zfill(4) + '.png')
    img = Image.open(img_path)  # H, W, C=3
    img = dataset.image_preprocess(img)
    images.append(img)

    if dataset.train_vae:
        # 训练VAE时，需将初始图像帧x0(images[0]), concat到每一个flow中（通道维）
        for i in range(dataset.seq_len):
            flows[i] = np.concatenate([images[0], flows[i]], axis=2)

    images = torch.as_tensor(images.copy()).permute(0, 3, 1, 2).contiguous().float()
    flows = torch.as_tensor(flows.copy()).permute(0, 3, 1, 2).contiguous().float()  # T C H W
    print(1)
