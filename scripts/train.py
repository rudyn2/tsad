import time

import torch
import wandb
from torch.utils.data import DataLoader, random_split
import pytorch_memlab
from pytorch_memlab import MemReporter


# @pytorch_memlab.profile
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
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    # val_loader = DataLoader(val, batch_size=int(batch_size / 8), shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    tiempo_epochs = 0
    global_step = 0
    best_loss = 1e100
    train_loss, train_acc, test_loss = [], [], []

    for e in range(1, epochs + 1):
        inicio_epoch = time.time()
        net.train()

        # Variables para las métricas
        running_loss, running_tl_acc, running_seg_acc, running_va_loss = 0.0, 0.0, 0.0, 0.0
        avg_tl_acc, avg_seg_acc, avg_va_loss, avg_loss = 0, 0, 0, 0

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
            avg_loss = running_loss / (i + 1)

            # accuracy of traffic lights
            _, max_idx = torch.max(y['traffic_light_status'], dim=1)
            running_tl_acc += torch.sum(max_idx == tl).item()
            avg_tl_acc = running_tl_acc / items * 100
            # accuracy semantic
            _, max_idx = torch.max(y['segmentation'], dim=1)
            running_seg_acc += torch.sum(max_idx == s).item() / max_idx.numel()
            avg_seg_acc = running_seg_acc / items * 100
            # error of vehicle affordances
            running_va_loss += torch.sum(((y['vehicle_affordances'] - v_aff).squeeze() * va_weights) ** 2).item()
            avg_va_loss = running_va_loss / items

            # report
            sys.stdout.write(f'\rEpoch:{e}({items}/{n_train}), '
                             + (f'lr:{lr_scheduler.get_last_lr()[0]:02.7f}, ' if lr_scheduler is not None else '')
                             + f'Loss:{avg_loss:02.5f}, '
                             + f'Train SEG Acc:{avg_seg_acc:02.1f}%, '
                             + f'Train TL Acc:{avg_tl_acc:02.1f}%, '
                             + f'Train VA Loss: {avg_va_loss:02.5f}')
            if use_wandb:
                wandb.log({'train/loss': float(avg_loss), 'train/acc TL': float(avg_tl_acc),
                           'train/acc SEG': float(avg_seg_acc), 'train/loss VA': float(avg_va_loss)}, step=global_step)
            global_step += 1

        tiempo_epochs += time.time() - inicio_epoch
        if use_wandb:
            wandb.log(
                {'train/loss': float(avg_loss), 'train/acc TL': float(avg_tl_acc), 'train/acc SEG': float(avg_seg_acc),
                 'train/loss VA': float(avg_va_loss), 'epoch': e})

        if e % reports_every == 0:
            sys.stdout.write(', Validating...')

            train_loss.append(avg_loss)
            train_acc.append([avg_tl_acc, avg_seg_acc, avg_va_loss])

            avg_tl_acc, avg_seg_acc, avg_va_loss = eval_net(device, net, seg_criterion, tl_criterion, va_criterion,
                                                            val_loader, va_weights)
            test_loss.append([avg_tl_acc, avg_seg_acc, avg_va_loss])
            sys.stdout.write(f', Val TL Acc:{avg_tl_acc:02.2f}%, '
                             + f'Val SEG Acc:{avg_seg_acc:02.2f}%, '
                             + f'Val VA Loss:{avg_va_loss:02.5f}%, '
                             + f'Avg-Time:{tiempo_epochs / e:.3f}s.\n')
            if use_wandb:
                wandb.log({'val/acc TL': float(avg_tl_acc), 'val/acc SEG': float(avg_seg_acc),
                           'val/loss VA': float(avg_va_loss)}, step=global_step)
                wandb.log({'val/acc TL': float(avg_tl_acc), 'val/acc SEG': float(avg_seg_acc),
                           'val/loss VA': float(avg_va_loss), 'epoch': e})

            # checkpointing
            if avg_loss <= best_loss:
                best_loss = avg_loss
                model_name = f"best_{net.__class__.__name__}_{e}.pth"
                torch.save(net.state_dict(), model_name)
                if use_wandb:
                    wandb.save(model_name)

        else:
            sys.stdout.write('\n')

        if lr_scheduler is not None:
            lr_scheduler.step()

    return train_loss, (train_acc, test_loss)


def eval_net(device, net, seg_criterion, tl_criterion, val_criterion, test_loader, va_weights):
    net.eval()
    running_tl_acc, running_seg_acc, running_va_loss = 0.0, 0.0, 0.0
    total_test = 0

    for i, (x, s, tl, v_aff) in enumerate(test_loader):
        x, s, tl, v_aff = x.to(device), s.to(device), tl.to(device), v_aff.to(device)

        with torch.no_grad():
            y = net(x)

        # l1 = seg_criterion(y['segmentation'], s)
        # l2 = tl_criterion(y['traffic_light_status'], tl)
        # l3 = val_criterion(y['vehicle_affordances'], v_aff)

        # accuracy of traffic lights
        _, max_idx = torch.max(y['traffic_light_status'], dim=1)
        running_tl_acc += torch.sum(max_idx == tl).item()
        # accuracy semantic
        _, max_idx = torch.max(y['segmentation'], dim=1)
        running_seg_acc += torch.sum(max_idx == s).item() / max_idx.numel()
        # error of vehicle affordances
        running_va_loss += torch.sum(((y['vehicle_affordances'] - v_aff).squeeze() * va_weights) ** 2).item()

        total_test += x.shape[0]

    avg_tl_acc = (running_tl_acc / total_test) * 100
    avg_seg_acc = (running_seg_acc / total_test) * 100
    avg_va_loss = running_va_loss / total_test
    return avg_tl_acc, avg_seg_acc, avg_va_loss


if __name__ == "__main__":
    import sys
    import torch.nn as nn
    import torch.optim as optim

    sys.path.append('..')
    from models import ADEncoder
    from models import HDF5Dataset
    from models import FocalLoss

    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed(0)

    # wandb.init(config=args, project="my-project")
    # wandb.init(project="tsad")
    # wandb.config["more"] = "custom"

    path = '../dataset'
    dataset = HDF5Dataset(path)
    model = ADEncoder()
    model.to(device)

    seg_loss = FocalLoss(apply_nonlin=torch.sigmoid)
    tl_loss_weights = torch.tensor([0.2, 0.8]).to(device)
    tl_loss = nn.BCEWithLogitsLoss(pos_weight=tl_loss_weights)
    va_loss = nn.MSELoss()                                                  # esta explotando

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_for_classification(model, dataset, optimizer,
                             seg_loss, tl_loss, va_loss,
                             criterion_weights=[1, 1, 1],
                             lr_scheduler=None,
                             epochs=1,
                             batch_size=1,
                             reports_every=1,
                             device=device,
                             val_percent=0.1,
                             va_weights=tl_loss_weights)
