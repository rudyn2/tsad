
import sys
sys.path.append('.')
sys.path.append('..')

from models.carlaEmbeddingDataset import CarlaEmbeddingDataset, CarlaOnlineEmbeddingDataset, PadSequence
from models.TemporalEncoder import RNNEncoder


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    import torch

    torch.cuda.empty_cache()

    device = 'cuda'
    d = CarlaOnlineEmbeddingDataset(embeddings_path='../embeddings.hdf5', json_path='../dataset/sample6.json')
    n_val = int(len(d) * 0.1)
    n_train = len(d) - n_val
    train, val = random_split(d, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=32, collate_fn=PadSequence())
    val_loader = DataLoader(val, batch_size=8, collate_fn=PadSequence())
    mse_loss = torch.nn.MSELoss()

    model = RNNEncoder(hidden_size=1028)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    step = 0
    for epoch in range(10):

        # Train
        train_total_loss = 0
        for embeddings, embeddings_length, actions, embeddings_label in train_loader:
            embeddings, embeddings_label, actions = embeddings.to(device), embeddings_label.to(device), actions.to(device)
            pred = model(embeddings, actions, embeddings_length)

            optimizer.zero_grad()
            loss = mse_loss(pred,  embeddings_label)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

        avg_train_loss = train_total_loss / len(train_loader)
        sys.stdout.write('\r')
        sys.stdout.write(f"Epoch: {epoch + 1} | Train loss: {avg_train_loss:.5f}")
        step += pred.shape[0]

        # Validate
        for embeddings, embeddings_length, actions, embeddings_label in train_loader:
            val_total_loss = 0
            with torch.no_grad():
                embeddings, embeddings_label, actions = embeddings.to(device), embeddings_label.to(device), actions.to(
                    device)
                pred = model(embeddings, actions, embeddings_length)
                loss = mse_loss(pred, embeddings_label)
                val_total_loss += loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        sys.stdout.write(f", Validation loss: {avg_val_loss:.5f}")
        sys.stdout.flush()
        sys.stdout.write('\n')


