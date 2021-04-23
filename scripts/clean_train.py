import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch


if __name__ == '__main__':

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

    path = '../dataset'
    dataset = HDF5Dataset(path)
    model = ADEncoder()
    model.to(device)

    seg_loss = FocalLoss(apply_nonlin=torch.sigmoid)
    tl_loss_weights = torch.tensor([0.2, 0.8]).to(device)
    tl_loss = nn.BCEWithLogitsLoss(pos_weight=tl_loss_weights)
    va_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=False)

    for e in range(1, 10 + 1):
        model.train()

        for i, (x, s, tl, v_aff) in enumerate(train_loader):

            # x, s, tl, v_aff = x.to(device), s.to(device), tl.to(device), v_aff.to(device)
            x = x.to(device)
            s = s.to(device)
            tl = tl.to(device)
            v_aff = v_aff.to(device)

            # optimization step
            optimizer.zero_grad()
            y = model(x)
            l1 = seg_loss(y['segmentation'], s)
            l2 = tl_loss(y['traffic_light_status'], tl)
            l3 = va_loss(y['vehicle_affordances'], v_aff)
            loss = l1 + l2 + l3
            loss.backward()
            optimizer.step()
            print(i)

