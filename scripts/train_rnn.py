
import sys
sys.path.append('.')
sys.path.append('..')

from models.carlaEmbeddingDataset import CarlaEmbeddingDataset, CarlaOnlineEmbeddingDataset, PadSequence
from models.TemporalEncoder import RNNEncoder
import argparse
import wandb


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    import torch

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--embeddings', default='../dataset/embeddings.hdf5', type=str, help='Path to embeddings hdf5')
    parser.add_argument('--metadata', default='../dataset/carla_dataset.json', type=str, help='Path to json file')
    parser.add_argument('--hidden-size', default=1028, type=int, help='LSTM hidden size')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='device')

    args = parser.parse_args()

    wandb.init(project='tsad', entity='autonomous-driving')

    device = args.device
    dataset = CarlaOnlineEmbeddingDataset(embeddings_path=args.embeddings, json_path=args.metadata)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=PadSequence())
    val_loader = DataLoader(val, batch_size=8, collate_fn=PadSequence())
    mse_loss = torch.nn.MSELoss()

    model = RNNEncoder(hidden_size=args.hidden_size)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    tag = ''    # tag = '*' if the model was saved in the last epoch
    best_val_loss = 1e100
    for epoch in range(100):

        # Train
        train_total_loss = 0
        for i, (embeddings, embeddings_length, actions, embeddings_label) in enumerate(train_loader):
            embeddings, embeddings_label, actions = embeddings.to(device), embeddings_label.to(device), actions.to(device)
            pred = model(embeddings, actions, embeddings_length)

            optimizer.zero_grad()
            loss = mse_loss(pred,  embeddings_label)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

            avg_train_loss = train_total_loss / len(train_loader)
            sys.stdout.write('\r')
            sys.stdout.write(f"{tag}Epoch: {epoch + 1}({i}/{len(train_loader)})| Train loss: {avg_train_loss:.5f}")
            wandb.log({'train/loss': avg_train_loss})

        # Validate
        for embeddings, embeddings_length, actions, embeddings_label in val_loader:
            val_total_loss = 0
            with torch.no_grad():
                embeddings, embeddings_label, actions = embeddings.to(device), embeddings_label.to(device), actions.to(
                    device)
                pred = model(embeddings, actions, embeddings_length)
                loss = mse_loss(pred, embeddings_label)
                val_total_loss += loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        wandb.log({'val/loss': avg_val_loss})

        # checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_name = f"best_{model.__class__.__name__}.pth"
            torch.save(model.state_dict(), model_name)
            wandb.save(model_name)
            tag = '*'

        sys.stdout.write(f", Validation loss: {avg_val_loss:.5f}")
        sys.stdout.flush()
        sys.stdout.write('\n')


