import argparse
import os
import wandb
import logging
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from model.VAE import *
from data.dataset import SceneFlowDataset


def train(net_Encoder,
          net_Sampler,
          net_Decoder,
          net_ImEncoder,
          args,
          on_device):
    # get dataloader
    train_loader, val_loader = load_data(args)

    # initialize logging
    experiment = wandb.init(project='Flow-VAE', resume='allow', anonymous='must')
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
    optimizer_netE = optim.Adam(net_Encoder.parameters(), lr=args.lr, eps=1e-8)
    optimizer_netS = optim.Adam(net_Sampler.parameters(), lr=args.lr, eps=1e-8)
    optimizer_netD = optim.Adam(net_Decoder.parameters(), lr=args.lr, eps=1e-8)
    optimizer_netI = optim.Adam(net_ImEncoder.parameters(), lr=args.lr, eps=1e-8)
    criterion = nn.MSELoss()
    # vgg19 = models.vgg19(pretrained=True)
    global_step = 0

    # train step
    for epoch in range(1, args.epochs + 1):
        net_Encoder.train()
        net_Sampler.train()
        net_Decoder.train()
        net_ImEncoder.train()
        epoch_loss = 0

        with tqdm.tqdm(total=n_train * args.seq_len, desc=f'Epoch {epoch}/{args.epochs}') as pbar:
            for batch in train_loader:
                image0 = batch['image0']
                true_flows = batch['flows']
                flows_add = batch['flows_add']
                image0 = image0.to(device=device, dtype=torch.float32)
                true_flows = true_flows.to(device=device, dtype=torch.float32)
                flows_add = flows_add.to(device=device, dtype=torch.float32)

                img_embed = net_ImEncoder(image0)
                mu, si = net_Encoder(flows_add)
                flow_embed = net_Sampler(mu, si)
                pred_flows = net_Decoder(img_embed, flow_embed)

                loss = criterion(pred_flows, true_flows)

                # updates
                optimizer_netE.zero_grad()
                optimizer_netS.zero_grad()
                optimizer_netD.zero_grad()
                optimizer_netI.zero_grad()
                loss.backward()
                optimizer_netE.step()
                optimizer_netS.step()
                optimizer_netD.step()
                optimizer_netI.step()

                global_step += 1
                epoch_loss += loss.item()
                pbar.update(args.batch_size)

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch,
                })

            pbar.set_postfix(**{'loss (batch)': epoch_loss})
            experiment.log({})

        # save model
        if epoch % 5 == 0:
            torch.save({
                'netE': net_Encoder,
                'netS': net_Sampler,
                'netD': net_Decoder,
                'netI': net_ImEncoder,
            },
                '%s/{epoch}.pth' % args.checkpoints)
            logging.info(f'No.{epoch} model saved.')


def get_args():
    parser = argparse.ArgumentParser(description='Train 3D VAE network')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path to load.')
    parser.add_argument('--image_dir', type=str, default='/media/tsy/I/scene flow/train_flow2rgb_test/images',
                        help='Directory to read PNG images.')
    parser.add_argument('--flows_dir', type=str, default='/media/tsy/I/scene flow/train_flow2rgb_test/flows',
                        help='Directory to read optical-flow file.')
    parser.add_argument('--seq_len', type=int, default=16, help='Number of images to be predicted.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for select samples.')
    parser.add_argument('--flow_prefix', type=str, default='OpticalFlowIntoFuture_', help='Prefix of flow file.')
    parser.add_argument('--flow_suffix', type=str, default='_L.pfm', help='Suffix of flow file.')
    parser.add_argument('--train_vae', type=bool, default=True, help='Indicator of training VAE or not.')
    parser.add_argument('--val_percent', type=float, default=0.1, help='Percentage of validation set.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.00002, help='Learning rate.')
    parser.add_argument('--beta', type=float, default=0.5, help='Momentum term for Adam.')
    parser.add_argument('--image_size', type=int, default=128, help='Image size.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_VAE', help='Directory to save models.')
    parser.add_argument('--lambda_', type=float, default=0.1, help='Weight to balance two losses.')
    parser.add_argument('--nf', type=int, default=2, help='Channel number of flows.')
    parser.add_argument('--nc', type=int, default=3, help='Channel number of image.')
    parser.add_argument('--z_dim', type=int, default=2000, help='Dimension of z.')
    parser.add_argument('--naf', type=int, default=64, help='Number of channels(for encoders).')
    parser.add_argument('--ngf', type=int, default=64, help='Number of channels(for decoder).')

    return parser.parse_args()


def load_data(args):
    # create dataset
    dataset = SceneFlowDataset(image_dir=args.image_dir,
                               flow_dir=args.flows_dir,
                               random_seed=args.seed,
                               seq_len=args.seq_len,
                               flow_prefix=args.flow_prefix,
                               flow_suffix=args.flow_suffix,
                               train_vae=args.train_vae,
                               image_size=args.image_size,
                               )
    # split dataset into train and validate
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # create data-loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':

    args_ = get_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args_.pretrained != '':
        pretrained = torch.load(args_.pretrained, map_location=device)
        netE = pretrained['netE']
        netS = pretrained['netS']
        netD = pretrained['netD']
        netI = pretrained['netI']
        print(f'Loaded models from file {args_.pretrained}.')
    else:
        # model definition
        netE = VolEncoder(channels=args_.nc + args_.nf, naf=args_.naf, z_dim=args_.z_dim)
        netS = VolSampler()
        netD = VolDecoder(channels=args_.nf, ngf=args_.ngf, z_dim=args_.z_dim)
        netI = ImEncoder(channels=args_.nc, naf=args_.naf, z_dim=args_.z_dim)

        netE.to(device=device)
        netS.to(device=device)
        netD.to(device=device)
        netI.to(device=device)
        print(f'Initialize models from scratch.')

    if not os.path.exists(args_.checkpoints):
        os.mkdir(args_.checkpoints)

    try:
        train(net_Encoder=netE,
              net_Sampler=netS,
              net_Decoder=netD,
              net_ImEncoder=netI,
              args=args_,
              on_device=device
              )

    except KeyboardInterrupt:
        torch.save({
            'netE': netE,
            'netS': netS,
            'netD': netD,
            'netI': netI,
        },
            '%s/Interrupt.pth' % args_.checkpoints)

