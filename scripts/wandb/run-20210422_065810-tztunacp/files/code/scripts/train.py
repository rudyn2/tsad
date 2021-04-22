import sys
import time

import torch
import wandb
from torch.utils.data import DataLoader, random_split


def train_for_classification(net, dataset, optimizer,
                            seg_criterion, tl_criterion, va_criterion,
                            criterion_weights=[1, 1, 1],
                            lr_scheduler=None,
                            epochs: int = 1,
                            batch_size: int = 64,
                            reports_every: int = 1,
                            device: str = 'cuda',
                            val_percent: float = 0.1):
    net.to(device)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=int(batch_size / 8), shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    tiempo_epochs = 0
    global_step = 0
    best_loss = 1e100
    train_loss, train_acc, test_loss = [], [], []

    for e in range(1, epochs + 1):
        inicio_epoch = time.time()
        net.train()

        # Variables para las métricas
        running_loss, running_acc = 0.0, 0.0
        avg_acc, avg_loss = 0, 0

        for i, (x, s, tl, v_aff) in enumerate(train_loader):

            x, s, tl, v_aff = x.to(device), s.to(device), tl.to(device), v_aff.to(device)
            
            # optimization step
            optimizer.zero_grad()
            y = net(x)
            l1 = seg_loss(y['segmentation'], s)
            l2 = tl_loss(y['traffic_light_status'], tl)
            l3 = va_loss(y['vehicle_affordances'], v_aff)
            loss = criterion_weights[0]*l1 + criterion_weights[1]*l2 + criterion_weights[2]*l3
            loss.backward()
            optimizer.step()

            # loss
            items = min(n_train, (i + 1) * train_loader.batch_size)
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            # accuracy of traffic lights
            _, max_idx = torch.max(y['traffic_light_status'], dim=1)
            running_acc += torch.sum(max_idx == tl).item()
            avg_acc = running_acc / items * 100

            # report
            sys.stdout.write(f'\rEpoch:{e}({items}/{n_train}), '
                             + (f'lr:{lr_scheduler.get_last_lr()[0]:02.7f}, ' if lr_scheduler is not None else '')
                             + f'Loss:{avg_loss:02.5f}, '
                             + f'Train TL Acc:{avg_acc:02.1f}%')
            wandb.log({'train/loss': float(avg_loss), 'train/acc TL': float(avg_acc)}, step=global_step)
            global_step += 1

        tiempo_epochs += time.time() - inicio_epoch
        wandb.log({'train/loss': float(avg_loss), 'train/acc TL': float(avg_acc), 'epoch': e})

        if e % reports_every == 0:
            sys.stdout.write(', Validating...')

            train_loss.append(avg_loss)
            train_acc.append(avg_acc)

            avg_loss = eval_net(device, net, val_loader, seg_criterion, tl_criterion, va_criterion, criterion_weights=criterion_weights)
            test_loss.append(avg_loss)
            sys.stdout.write(f', Val Acc:{avg_loss:02.2f}%, '
                             + f'Avg-Time:{tiempo_epochs / e:.3f}s.\n')
            wandb.log({'val/acc': float(avg_loss)}, step=global_step)
            wandb.log({'val/acc': float(avg_loss), 'epoch': e})

            # checkpointing
            if avg_loss <= best_loss:
                best_loss = avg_loss
                model_name = f"best_{net.__class__.__name__}_{e}.pth"
                torch.save(net.state_dict(), model_name)
                wandb.save(model_name)

        else:
            sys.stdout.write('\n')

        if lr_scheduler is not None:
            lr_scheduler.step()

    return train_loss, (train_acc, test_loss)


def eval_net(device, net, test_loader,
                seg_criterion, tl_criterion, va_criterion,
                criterion_weights=[1, 1, 1]):
    net.eval()
    running_loss = 0.0
    total_test = 0

    for i, (x, s, tl, v_aff) in enumerate(test_loader):
        x, s, tl, v_aff = x.to(device), s.to(device), tl.to(device), v_aff.to(device)

        l1 = seg_loss(y['segmentation'], s)
        l2 = tl_loss(y['traffic_light_status'], tl)
        l3 = va_loss(y['vehicle_affordances'], v_aff)

        running_loss += criterion_weights[0]*l1 + criterion_weights[1]*l2 + criterion_weights[2]*l3
        total_test += len(labels)

    avg_loss = (running_loss / total_test) * 100
    return avg_loss



if __name__ == "__main__":
    import sys
    import torch.nn as nn
    import torch.optim as optim
    from pathlib import Path
    sys.path.append('..')
    from models import ADEncoder
    from models import HDF5Dataset
    from models import FocalLoss

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed(0)
    
    #wandb.init(config=args, project="my-project")
    wandb.init(project="tsad")
    wandb.config["more"] = "custom"
    
    path = '../dataset'
    dataset = HDF5Dataset(path)
    model = ADEncoder().to(device)

    seg_loss = FocalLoss(apply_nonlin=torch.sigmoid)
    tl_loss_weights = torch.tensor([0.2, 0.8]).to(device)
    tl_loss = nn.BCEWithLogitsLoss(pos_weight=tl_loss_weights)
    va_loss = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_for_classification(model, dataset, optimizer,
                            seg_loss, tl_loss, va_loss,
                            criterion_weights=[1, 1, 1],
                            lr_scheduler=None,
                            epochs = 1,
                            batch_size = 1,
                            reports_every = 1,
                            device = device,
                            val_percent = 0.1)