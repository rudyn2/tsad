import sys
import time

import torch
import wandb
from torch.utils.data import DataLoader, random_split


def train_for_classification(net, dataset, optimizer,
                             criterion, lr_scheduler=None,
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
    val_loader = DataLoader(val, batch_size=int(batch_size / 8), shuffle=False, num_workers=2, pin_memory=True,
                            drop_last=True)

    tiempo_epochs = 0
    global_step = 0
    best_acc = 0
    train_loss, train_acc, test_acc = [], [], []

    for e in range(1, epochs + 1):
        inicio_epoch = time.time()
        net.train()

        # Variables para las métricas
        running_loss, running_acc = 0.0, 0.0
        avg_acc, avg_loss = 0, 0

        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # optimization step
            optimizer.zero_grad()
            logits = net(images)
            y_pred = logits.type(torch.DoubleTensor)  # probability distribution over classes
            labels = labels.type(torch.LongTensor)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            # loss
            items = min(n_train, (i + 1) * train_loader.batch_size)
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            # accuracy
            _, max_idx = torch.max(y_pred, dim=1)
            running_acc += torch.sum(max_idx == labels).item()
            avg_acc = running_acc / items * 100

            # report
            sys.stdout.write(f'\rEpoch:{e}({items}/{n_train}), '
                             + (f'lr:{lr_scheduler.get_last_lr()[0]:02.7f}, ' if lr_scheduler is not None else '')
                             + f'Loss:{avg_loss:02.5f}, '
                             + f'Train Acc:{avg_acc:02.1f}%')
            wandb.log({'train/loss': float(avg_loss), 'train/acc': float(avg_acc)}, step=global_step)
            global_step += 1

        tiempo_epochs += time.time() - inicio_epoch
        wandb.log({'train/loss': float(avg_loss), 'train/acc': float(avg_acc), 'epoch': e})

        if e % reports_every == 0:
            sys.stdout.write(', Validating...')

            train_loss.append(avg_loss)
            train_acc.append(avg_acc)

            avg_acc = eval_net(device, net, val_loader)
            test_acc.append(avg_acc)
            sys.stdout.write(f', Val Acc:{avg_acc:02.2f}%, '
                             + f'Avg-Time:{tiempo_epochs / e:.3f}s.\n')
            wandb.log({'val/acc': float(avg_acc)}, step=global_step)
            wandb.log({'val/acc': float(avg_acc), 'epoch': e})

            # checkpointing
            if avg_acc >= best_acc:
                best_acc = avg_acc
                model_name = f"best_{net.__class__.__name__}_{e}.pth"
                torch.save(net.state_dict(), model_name)
                wandb.save(model_name)

        else:
            sys.stdout.write('\n')

        if lr_scheduler is not None:
            lr_scheduler.step()

    return train_loss, (train_acc, test_acc)


def eval_net(device, net, test_loader):
    net.eval()
    running_acc = 0.0
    total_test = 0

    for i, data in enumerate(test_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        logits = net(images.float())
        _, max_idx = torch.max(logits, dim=1)
        running_acc += torch.sum(max_idx == labels).item()
        total_test += len(labels)

    avg_acc = (running_acc / total_test) * 100
    return avg_acc