import sys

sys.path.append('.')
sys.path.append('..')

from models.carlaEmbeddingDataset import CarlaOnlineEmbeddingDataset, PadSequence, CarlaEmbeddingDataset
from models.TemporalEncoder import RNNEncoder, VanillaRNNEncoder
import argparse
import wandb

if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    import torch

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--embeddings', default='../dataset/embeddings.hdf5', type=str, help='Path to embeddings hdf5')
    parser.add_argument('--metadata', default='../dataset/carla_dataset_repaired.json', type=str, help='Path to json file')
    parser.add_argument('--dataset', default='online', type=str, help='Type of dataset. online: all the dataset will'
                                                                      'be loaded into memory before training. '
                                                                      'offline: the embeddings'
                                                                      'will be loaded lazily')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional LSTM or not.')
    parser.add_argument('--hidden-size', default=256, type=int, help='LSTM hidden size')
    parser.add_argument('--num-layers', default=2, type=int, help='LSTM number of hidden layers')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--val-size', default=0.1, type=float,
                        help='Ratio of train dataset that will be used for validation')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    
    parser.add_argument('--action-channels', default=128, type=int, help='Number of channels in action codification')
    parser.add_argument('--speed-channels', default=128, type=int, help='Number of channels in speed codification')
    parser.add_argument('--state-channels', default=2048, type=int, help='Number of channels in state codification, only used in vanilla')
    parser.add_argument('--rnn-model', default='vanilla', type=str, help='Which rnn model use: "vanilla" or "convol"')

    args = parser.parse_args()

    wandb.init(project='tsad', entity='autonomous-driving')

    device = args.device
    if args.dataset == 'online':
        dataset = CarlaOnlineEmbeddingDataset(embeddings_path=args.embeddings, json_path=args.metadata)
    else:
        dataset = CarlaEmbeddingDataset(embeddings_path=args.embeddings, json_path=args.metadata)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=PadSequence(), shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=PadSequence())
    mse_loss = torch.nn.MSELoss()

    if args.rnn_model == "vanilla":
        model = VanillaRNNEncoder(
            num_layers=args.num_layers, 
            hidden_size=args.hidden_size,
            action__chn=args.action_channels,
            speed_chn=args.speed_channels,
            state_chn=args.state_channels
            )
    elif args.rnn_model == "convol":
        model = RNNEncoder(
            num_layers=args.num_layers, 
            hidden_size=args.hidden_size,
            action__chn=args.action_channels,
            speed_chn=args.speed_channels
        )
    else:
        raise ValueError('Model not implemented')
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = wandb.config
    config.model = model.__class__.__name__
    config.device = device
    config.batch_size = args.batch_size
    config.hidden_size = args.hidden_size
    config.epochs = args.epochs
    config.learning_rate = args.lr

    tag = ''  # tag = '*' if the model was saved in the last epoch
    best_val_loss = 1e100
    for epoch in range(args.epochs):

        # Train
        model.train()
        train_total_loss = 0
        for i, (embeddings, embeddings_length, actions, speeds, embeddings_label) in enumerate(train_loader):
            embeddings, embeddings_label, actions, speeds = embeddings.to(device), embeddings_label.to(device), actions.to(
                device), speeds.to(device)
            pred = model(embeddings, actions, speeds, embeddings_length)

            optimizer.zero_grad()
            loss = mse_loss(pred, embeddings_label)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

            avg_train_loss = train_total_loss / (i + 1)
            sys.stdout.write('\r')
            sys.stdout.write(f"{tag}Epoch: {epoch + 1}({i}/{len(train_loader)})| Train loss: {avg_train_loss:.5f}")
            wandb.log({'train/loss': avg_train_loss})

        avg_train_loss = train_total_loss / len(train_loader)
        wandb.log({'train/loss': avg_train_loss, 'epoch': epoch + 1})

        # Validate
        val_total_loss = 0
        model.eval()
        for embeddings, embeddings_length, actions, speeds, embeddings_label in val_loader:
            with torch.no_grad():
                embeddings, embeddings_label, actions, speeds = embeddings.to(device), embeddings_label.to(device), actions.to(
                    device), speeds.to(device)
                pred = model(embeddings, actions, speeds, embeddings_length)
                loss = mse_loss(pred, embeddings_label)
                val_total_loss += loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        wandb.log({'val/loss': avg_val_loss, 'epoch': epoch + 1})

        # checkpointing
        tag = ''
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_name = f"best_{model.__class__.__name__}.pth"
            torch.save(model.state_dict(), model_name)
            # wandb.save(model_name)
            tag = '*'

        sys.stdout.write(f", Validation loss: {avg_val_loss:.5f}")
        sys.stdout.flush()
        sys.stdout.write('\n')
    print("Finished!")
