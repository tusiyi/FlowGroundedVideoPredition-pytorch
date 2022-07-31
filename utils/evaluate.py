import torch
import torch.nn as nn

from tqdm import tqdm


def evaluate(netC, netF, netD, val_loader, device, args):
    netC.eval()
    netF.eval()
    netD.eval()
    criterion = nn.MSELoss()

    num_val_batches = len(val_loader)
    n_val = num_val_batches * args.batch_size
    total_loss = 0

    with tqdm(total=n_val, desc='Validation', unit='img') as pbar:
        for batch in val_loader:
            images = batch['images']
            flows = batch['flows']
            images = images.to(device=device, dtype=torch.float32)
            flows = flows.to(device=device, dtype=torch.float32)
            image0 = images[:, 0, ...]
            batch_loss = 0
            with torch.no_grad():
                for i in range(args.seq_len):
                    true_image = images[:, i + 1, ...]
                    img_feat = netC(image0)
                    flow_feat = netF(flows[:, i, ...])
                    pred_image = netD(img_feat, flow_feat)
                    # reconstruction loss
                    loss = criterion(pred_image, true_image)
                    batch_loss += loss.item()

                    image0 = pred_image

            total_loss += batch_loss
            pbar.update(args.batch_size)

    if num_val_batches == 0:
        return total_loss
    else:
        return total_loss / num_val_batches  # batch average loss
