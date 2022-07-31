import argparse
import os
import wandb
import logging
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torchvision.models as models

import model.vgg_conv4_1 as vgg
from data.dataset import SceneFlowDataset
from utils.vgg19 import get_features
from utils.evaluate import evaluate


def feature_loss(net, true_img, pred_img):
    feat_loss = 0
    with torch.no_grad():
        t_feats = get_features(net, true_img)
        p_feats = get_features(net, pred_img)

    base_loss = nn.MSELoss()
    for j in range(5):
        feat_loss += base_loss(p_feats[j], t_feats[j])

    return feat_loss


def train(net_Content,
          net_FLow,
          net_Decode,
          net_vgg,
          args,
          on_device):
    # get dataloader
    train_loader, val_loader = load_data(args)

    # initialize logging
    experiment = wandb.init(project='Flow2RGB', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=args.epochs, batch_size=args.batch_size,
                                  learning_rate=args.lr, val_percent=args.val_percent,
                                  image_size=args.image_size, sequence_length=args.seq_len,
                                  random_seed=args.seed,))

    # log info
    n_train = len(train_loader) * args.batch_size
    n_val = len(val_loader) * args.batch_size

    logging.info(f"""
    Epochs:          {args.epochs}
    Batch size:      {args.batch_size}
    Learning rate:   {args.lr}
    Training size:   {n_train}
    Validation size: {n_val}
    Device:          {on_device.type}
    Images size:     {args.image_size}
    """)

    # optimizers and loss
    optimizer_netC = optim.Adam(net_Content.prameters(), lr=args.lr, eps=1e-8)
    optimizer_netF = optim.Adam(net_FLow.parameters(), lr=args.lr, eps=1e-8)
    optimizer_netD = optim.Adam(net_Decode.parameters(), lr=args.lr, eps=1e-8)
    criterion = nn.MSELoss()
    # vgg19 = models.vgg19(pretrained=True)
    global_step = 0

    # train step
    for epoch in range(args.epochs + 1):
        net_Content.train()
        net_FLow.train()
        net_Decode.train()
        epoch_loss = 0

        with tqdm.tqdm(total=n_train * args.seq_len, desc=f'Epoch {epoch}/{args.epochs}') as pbar:
            for batch in train_loader:
                seq_loss = 0
                images = batch['images']
                flows = batch['flows']
                images = images.to(device=device, dtype=torch.float32)
                flows = flows.to(device=device, dtype=torch.float32)
                image0 = images[:, 0, ...]
                # multiple steps for sequence
                for i in range(args.seq_len):
                    true_image = images[:, i + 1, ...]
                    img_feat = net_Content(image0)
                    flow_feat = net_FLow(flows[:, i, ...])
                    pred_image = net_Decode(img_feat, flow_feat)
                    # reconstruction loss
                    loss = criterion(pred_image, true_image)
                    # feature loss
                    true_feats = get_features(net_vgg, true_image)
                    pred_feats = get_features(net_vgg, pred_image)
                    for j in range(5):
                        loss += criterion(pred_feats[j], true_feats[j])
                    # update
                    optimizer_netC.zero_grad()
                    optimizer_netF.zero_grad()
                    optimizer_netD.zero_grad()
                    loss.backward()
                    optimizer_netC.step()
                    optimizer_netF.step()
                    optimizer_netD.step()

                    global_step += 1
                    seq_loss += loss.item()
                    pbar.update(args.batch_size)
                    # update image0
                    image0 = pred_image

                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch,
                    })

                pbar.set_postfix(**{'loss (batch)': seq_loss})
                experiment.log({})

                # validating step (only reconstruction loss)
                val_loss = evaluate(net_Content, net_FLow, net_Decode, val_loader, device, args)
                experiment.log({
                    'train loss(batch)': seq_loss,
                    'validate loss(batch)': val_loss,
                    'step': global_step,
                    'epoch': epoch,
                    'true_image(last frame)': wandb.Image(true_image[0].float().cpu()),
                    'pred_image(last frame)': wandb.Image(pred_image[0].float().cpu()),
                })
        # save model
        if epoch % 5 == 0:
            torch.save({
                'netC': netC,
                'netF': netF,
                'netD': netD,
            },
                '%s/{epoch}.pth' % args.checkpoints)
            logging.info(f'No.{epoch} model saved.')


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

    return parser


def load_data(args):
    # create dataset
    dataset = SceneFlowDataset(image_dir=args.image_dir,
                               flow_dir=args.flows_dir,
                               random_seed=args.seed,
                               seq_len=args.seq_len,
                               flow_prefix=args.flow_prefix,
                               flow_suffix=args.flow_suffix,
                               train_vae=args.train_vae,
                               )
    # split dataset into train and validate
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # create dataloaders
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':

    args_ = get_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args_.pretrained != '':
        pretrained = torch.load(args_.pretrained, map_location=device)
        netC = pretrained['netC']
        netF = pretrained['netF']
        netD = pretrained['netD']
        print(f'Loaded models from file {args_.pretrained}.')
    else:
        # model definition
        netC = vgg.Conv4_1(in_channels=3, channels=256)
        netF = vgg.Conv4_1(in_channels=2, channels=256)
        netD = vgg.InvertConcatConv4_1(channels=256, out_channels=3)

        netC.to(device=device)
        netF.to(device=device)
        netD.to(device=device)
        print(f'Initialize models from scratch.')

    vgg19 = models.vgg19(pretrained=True)
    vgg19.to(device=device)

    if not os.path.exists(args_.checkpoints):
        os.mkdir(args_.checkpoints)

    try:
        train(net_Content=netC,
              net_FLow=netF,
              net_Decode=netD,
              net_vgg=vgg19,
              args=args_,
              on_device=device,
              )

    except KeyboardInterrupt:
        torch.save({
            'netC': netC,
            'netF': netF,
            'netD': netD,
                },
            '%s/Interrupt.pth' % args_.checkpoints)
