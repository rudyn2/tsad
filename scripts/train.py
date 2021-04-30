import time

import torch
import wandb
from torch.utils.data import DataLoader, random_split


def train_for_classification(net, dataset, optimizer,
                             seg_criterion, tl_criterion, va_criterion,
                             criterion_weights=(1 / 3, 1 / 3, 1 / 3),
                             lr_scheduler=None,
                             epochs: int = 1,
                             batch_size: int = 64,
                             reports_every: int = 1,
                             device: torch.device = 'cuda',
                             val_percent: float = 0.1,
                             use_wandb=False,
                             va_weights=None):
    if va_weights is None:
        va_weights = torch.Tensor([0.5, 0.5]).to(device)
    net.to(device)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=int(batch_size / 8),
    # shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    tiempo_epochs = 0
    global_step = 0
    best_loss = 1e100
    train_loss, train_acc, test_loss = [], [], []

    for e in range(1, epochs + 1):
        inicio_epoch = time.time()
        net.train()

        # Variables para las métricas
        running_tl_acc, running_seg_acc = 0.0, 0.0
        running_seg_loss, running_tl_loss, running_va_loss, running_loss = 0.0, 0.0, 0.0, 0.0
        avg_tl_acc, avg_seg_acc, avg_va_loss, avg_loss, avg_tl_loss, avg_seg_loss = 0, 0, 0, 0, 0, 0

        for i, (x, s, tl, v_aff) in enumerate(train_loader):

            # x, s, tl, v_aff = x.to(device), s.to(device), tl.to(device), v_aff.to(device)
            x = x.to(device)
            s = s.to(device)
            tl = tl.to(device)
            v_aff = v_aff.to(device)

            # optimization step
            optimizer.zero_grad()
            y = net(x)
            l1 = seg_criterion(y['segmentation'], s)
            l2 = tl_criterion(y['traffic_light_status'], tl)
            l3 = va_criterion(y['vehicle_affordances'], v_aff)
            loss = criterion_weights[0] * l1 + criterion_weights[1] * l2 + criterion_weights[2] * l3
            loss.backward()
            optimizer.step()

            # loss
            items = min(n_train, (i + 1) * train_loader.batch_size)
            running_loss += loss.item()
            running_seg_loss += l1.item()
            running_tl_loss += l2.item()
            running_va_loss += l3.item()

            # averaging losses
            avg_loss = running_loss / (i + 1)
            avg_seg_loss = running_seg_loss / (i + 1)
            avg_tl_loss = running_tl_loss / (i + 1)
            avg_va_loss = running_va_loss / (i + 1)

            # accuracy of traffic lights
            _, max_idx = torch.max(y['traffic_light_status'], dim=1)
            running_tl_acc += torch.sum(max_idx == torch.argmax(tl, dim=1)).item()
            avg_tl_acc = running_tl_acc / items * 100

            # accuracy semantic
            _, max_idx = torch.max(y['segmentation'], dim=1)
            running_seg_acc += torch.sum(max_idx == s).item() / max_idx.numel()
            avg_seg_acc = running_seg_acc / items * 100

            # report
            sys.stdout.write(f'\rEpoch:{e}({items}/{n_train}), '
                             + (f'lr:{lr_scheduler.get_last_lr()[0]:02.7f}, ' if lr_scheduler is not None else '')
                             + f'Train[Loss:{avg_loss:02.5f}, '
                             + f'SEG Acc:{avg_seg_acc:02.1f}%, '
                             + f'TL Acc:{avg_tl_acc:02.1f}%, '
                             + f'VA Loss: {avg_va_loss:02.5f}]')
            if use_wandb:
                wandb.log({'train/loss': float(avg_loss), 'train/acc TL': float(avg_tl_acc),
                           'train/loss SEG': float(avg_seg_loss), 'train/loss TL': float(avg_tl_loss),
                           'train/acc SEG': float(avg_seg_acc), 'train/loss VA': float(avg_va_loss)}, step=global_step)
            global_step += 1

        tiempo_epochs += time.time() - inicio_epoch
        if use_wandb:
            wandb.log(
                {'train/loss': float(avg_loss),
                 'train/acc TL': float(avg_tl_acc),
                 'train/acc SEG': float(avg_seg_acc),
                 'train/loss VA': float(avg_va_loss),
                 'train/loss TL': float(avg_tl_loss),
                 'train/loss SEG': float(avg_seg_loss),
                 'epoch': e})

        if e % reports_every == 0:
            sys.stdout.write(', Validating...')

            train_loss.append(avg_loss)
            train_acc.append([avg_tl_acc, avg_seg_acc, avg_va_loss])

            avg_tl_acc, avg_seg_acc, avg_loss, avg_seg_loss, avg_tl_loss, avg_va_loss = eval_net(device, net,
                                                                                                 seg_criterion,
                                                                                                 tl_criterion,
                                                                                                 va_criterion,
                                                                                                 val_loader, va_weights)
            test_loss.append([avg_tl_acc, avg_seg_acc, avg_va_loss])
            sys.stdout.write(f', Val[Loss:{avg_loss:02.4f}, '
                             + f'TL Acc:{avg_tl_acc:02.2f}%, '
                             + f'SEG Acc:{avg_seg_acc:02.2f}%, '
                             + f'VA Loss:{avg_va_loss:02.5f}%, '
                             + f'Avg-Time:{tiempo_epochs / e:.3f}s.\n')
            if use_wandb:
                wandb.log({'val/acc TL': float(avg_tl_acc), 'val/acc SEG': float(avg_seg_acc),
                           'val/loss VA': float(avg_va_loss), 'val/loss': float(avg_loss),
                           'val/loss TL': float(avg_tl_loss), 'val/loss SEG': float(avg_seg_loss)},
                          step=global_step)
                wandb.log({'val/acc TL': float(avg_tl_acc), 'val/acc SEG': float(avg_seg_acc),
                           'val/loss VA': float(avg_va_loss), 'val/loss': float(avg_loss),
                           'val/loss TL': float(avg_tl_loss), 'val/loss SEG': float(avg_seg_loss), 'epoch': e})

            # checkpointing
            if avg_loss <= best_loss:
                best_loss = avg_loss
                model_name = f"best_{net.__class__.__name__}.pth"
                torch.save(net.state_dict(), model_name)
                if use_wandb:
                    wandb.save(model_name)
                    wandb.log({'best_val_multitask_loss': float(best_loss), 'epoch': e})

        else:
            sys.stdout.write('\n')

        if lr_scheduler is not None:
            lr_scheduler.step()

    return train_loss, (train_acc, test_loss)


def eval_net(device, net, seg_criterion, tl_criterion, val_criterion, test_loader, va_weights):
    net.eval()
    running_tl_acc, running_seg_acc, running_va_loss, running_loss = 0.0, 0.0, 0.0, 0.0
    running_seg_loss, running_tl_loss = 0.0, 0.0
    total_test = 0
    avg_loss, avg_seg_loss, avg_tl_loss, avg_va_loss = 0, 0, 0, 0

    for i, (x, s, tl, v_aff) in enumerate(test_loader):
        x, s, tl, v_aff = x.to(device), s.to(device), tl.to(device), v_aff.to(device)

        with torch.no_grad():
            y = net(x)

        l1 = seg_criterion(y['segmentation'], s)
        l2 = tl_criterion(y['traffic_light_status'], tl)
        l3 = val_criterion(y['vehicle_affordances'], v_aff)
        loss = l1 + l2 + l3

        running_loss += loss.item()
        running_seg_loss += l1.item()
        running_tl_loss += l2.item()
        running_va_loss += l3.item()

        # averaging losses
        avg_loss = running_loss / (i + 1)
        avg_seg_loss = running_seg_loss / (i + 1)
        avg_tl_loss = running_tl_loss / (i + 1)
        avg_va_loss = running_va_loss / (i + 1)

        # accuracy of traffic lights
        _, max_idx = torch.max(y['traffic_light_status'], dim=1)
        running_tl_acc += torch.sum(max_idx == torch.argmax(tl, dim=1)).item()
        # accuracy semantic
        _, max_idx = torch.max(y['segmentation'], dim=1)
        running_seg_acc += torch.sum(max_idx == s).item() / max_idx.numel()

        total_test += x.shape[0]

    avg_tl_acc = (running_tl_acc / total_test) * 100
    avg_seg_acc = (running_seg_acc / total_test) * 100

    return avg_tl_acc, avg_seg_acc, avg_loss, avg_seg_loss, avg_tl_loss, avg_va_loss


if __name__ == "__main__":
    import sys
    import torch.nn as nn
    import torch.optim as optim

    sys.path.append('..')
    sys.path.append('.')
    from models.ADEncoder import ADEncoder
    from models.carlaDataset import HDF5Dataset
    from models.carlaDatasetSimple import CarlaDatasetSimple
    from models.losses import FocalLoss, WeightedPixelWiseNLLoss
    import argparse

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../dataset', type=str, help='Path to dataset folder')
    parser.add_argument('--dataset', default='simple', type=str, help='Type of dataset [cached, simple]')
    parser.add_argument('--cache-size', default=1024, type=int, help='Cache size of cached dataset.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size.')
    parser.add_argument('--use-bn', action="store_true",
                        help='Whether to use batch normalization at upconvolution layers or not.')
    parser.add_argument('--backbone-type', default="efficientnet", type=str,
                        help='Backbone architecture [resnet, efficientnet-b[0-7]].')
    parser.add_argument('--loss-weights', default="1, 1, 1", type=str,
                        help='Loss weights [segmentation, traffic light status, vehicle affordances ]')
    parser.add_argument('--tl-weights', default="0.2, 0.8", type=str,
                        help='Traffic light weights [Green, Red]')
    parser.add_argument('--loss-fn', default='focal-loss', type=str, help='Loss function [focal-loss, wce] used for '
                                                                          'semantic segmentation')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate.')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed(0)

    wandb.init(project='tsad', entity='autonomous-driving')

    print("Loading data")
    if args.dataset == 'cached':
        dataset = HDF5Dataset(args.data)
    else:
        dataset = CarlaDatasetSimple(args.data)

    model = ADEncoder(backbone=args.backbone_type, use_bn=args.use_bn)
    model.to(device)

    tl_weights = str(args.tl_weights).split(",")
    tl_weights = [float(s) for s in tl_weights]

    if args.loss_fn == 'focal-loss':
        seg_loss = FocalLoss(apply_nonlin=torch.sigmoid)
    else:
        # moving obstacles (0),  traffic lights (1),  road markers(2),  road (3),  sidewalk (4) and background (5).
        seg_loss = WeightedPixelWiseNLLoss(weights={
            0: 0.5,
            1: 0.1,
            2: 0.15,
            3: 0.1,
            4: 0.1,
            5: 0.05

        })

    tl_loss_weights = torch.tensor(tl_weights).to(device)
    tl_loss = nn.BCEWithLogitsLoss(pos_weight=tl_loss_weights)
    va_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # wandb config specification
    config = wandb.config
    config.type_dataset = args.dataset
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.model = args.backbone_type
    config.loss_weights = args.loss_weights
    config.tl_weights = args.tl_weights
    config.segmentation_loss = args.loss_fn
    config.optimizer = optimizer.__class__.__name__

    print("Training...")
    loss_weights = str(args.loss_weights).split(",")
    loss_weights = [float(s) for s in loss_weights]
    train_for_classification(model, dataset, optimizer,
                             seg_loss, tl_loss, va_loss,
                             criterion_weights=loss_weights,
                             lr_scheduler=None,
                             epochs=args.epochs,
                             batch_size=args.batch_size,
                             reports_every=1,
                             device=device,
                             val_percent=0.1,
                             va_weights=tl_loss_weights,
                             use_wandb=True)
