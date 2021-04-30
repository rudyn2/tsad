from ray import tune
from train import train_for_classification


def train(config: dict):
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


    train_for_classification()
    pass


if __name__ == '__main__':
    analysis = tune.run(
        train,
        config={
            "alpha": tune.grid_search([0.001, 0.01, 0.1]),
            "beta": tune.choice([1, 2, 3])
        })

    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))
