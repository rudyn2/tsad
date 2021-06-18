from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, RootMeanSquaredError, DiceCoefficient, ConfusionMatrix, \
    MetricsLambda, IoU
from termcolor import colored
from tqdm import tqdm
from ignite.utils import to_onehot
from torch.utils.data import random_split, DataLoader
from ignite.contrib.handlers.wandb_logger import WandBLogger, global_step_from_engine
from custom_metrics import Recall
from ignite.handlers import EarlyStopping, ModelCheckpoint


def prepare_batch(batch, device, non_blocking):
    y_expected = {}
    for t, label in zip(batch[1:], ["segmentation", "traffic_light_status", "vehicle_affordances", "pedestrian"]):
        y_expected[label] = t.to(device=device, non_blocking=non_blocking)
    x = batch[0].to(device=device, non_blocking=non_blocking)
    return x, y_expected


def output_transform_tl(process_output):
    """
    Output transform for traffic light status metrics.
    """
    y_pred = process_output[0]['traffic_light_status'].argmax(dim=1)
    y = process_output[1]['traffic_light_status'].argmax(dim=1)
    return dict(y_pred=y_pred, y=y)  # output format is according to `Accuracy` docs


def output_transform_ped(process_output):
    """
    Output transform for pedestrian presence metrics.
    """
    y_pred = process_output[0]['pedestrian'].argmax(dim=1)
    y = process_output[1]['pedestrian'].argmax(dim=1)
    return dict(y_pred=y_pred, y=y)  # output format is according to `Accuracy` docs


def output_transform_va(process_output):
    """
    Output transform for vehicle affordances metrics.
    """
    y_pred = process_output[0]['vehicle_affordances'].argmax(dim=1)
    y = process_output[1]['vehicle_affordances'].argmax(dim=1)
    return dict(y_pred=y_pred, y=y)  # output format is according to `MeanSquareError` docs


def output_transform_seg(process_output):
    """
    Output transform for segmentation metrics.
    """

    y_pred = process_output[0]['segmentation'].argmax(dim=1)  # (B, W, H)
    y = process_output[1]['segmentation']  # (B, W, H)
    y_pred_ = y_pred.view(-1)  # B, (W*H)
    y_ = y.view(-1)
    y_pred_one_hot = to_onehot(y_pred_, num_classes=6)
    return dict(y_pred=y_pred_one_hot, y=y_)  # output format is according to `DiceCoefficient` docs


def run(args):
    # init data loaders
    log_interval = 1
    print(colored("[*] Initializing dataset and dataloader", "white"))
    dataset = CarlaDatasetSimple(args.data)
    n_val = int(len(dataset) * 0.05)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    print(colored("[+] Dataset & Dataloader Ready!", "green"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(colored("Using device: ", "white") + colored(device, "green"))

    print(colored("[*] Initializing model, optimizer and loss", "white"))
    model = ADEncoder(backbone='efficientnet-b5')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = ADLoss(loss_weights=args.loss_weights,
                  tl_weights=args.tl_weights,
                  pd_weights=args.pd_weights)
    print(colored("[+] Model, optimizer and loss are ready!", "green"))

    avg_fn = lambda x: torch.mean(x).item()
    cm_metric = ConfusionMatrix(num_classes=6, output_transform=output_transform_seg)
    metrics = {
        'loss': Loss(loss_fn=loss, output_transform=lambda x: (x[0], x[1])),
        'loss_avg': RunningAverage(Loss(loss_fn=loss, output_transform=lambda x: (x[0], x[1]))),
        'tl_accuracy': Accuracy(output_transform=output_transform_tl),
        'tl_recall': Recall(output_transform=output_transform_tl),
        'tl_accuracy_avg': RunningAverage(Accuracy(output_transform=output_transform_tl)),
        'tl_recall_avg': RunningAverage(Recall(output_transform=output_transform_tl)),
        'pd_accuracy': Accuracy(output_transform=output_transform_ped),
        'pd_recall': Recall(output_transform=output_transform_ped),
        'pd_accuracy_avg': RunningAverage(Accuracy(output_transform=output_transform_ped)),
        'pd_recall_avg': RunningAverage(Recall(output_transform=output_transform_ped)),
        'va_rmse': RootMeanSquaredError(output_transform=output_transform_va),
        'va_rmse_avg': RunningAverage(RootMeanSquaredError(output_transform=output_transform_va)),
        'seg_dice': MetricsLambda(avg_fn, DiceCoefficient(cm_metric, ignore_index=5)),
        # 'seg_dice_avg': RunningAverage(MetricsLambda(avg_fn, DiceCoefficient(cm_metric, ignore_index=5))),
        'seg_iou': MetricsLambda(avg_fn, IoU(cm_metric, ignore_index=5)),
        # 'seg_iou_avg': RunningAverage(MetricsLambda(avg_fn, IoU(cm_metric, ignore_index=5)))
    }
    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        loss,
                                        prepare_batch=prepare_batch,
                                        output_transform=lambda x, y, y_pred, loss: (y_pred, y, loss),
                                        device=device)

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, prepare_batch=prepare_batch)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, prepare_batch=prepare_batch)

    # add metrics to trainer engine
    for label, metric in metrics.items():
        metric.attach(trainer, label, "batch_wise")

    score_function = lambda engine: -engine.state.metrics['loss']
    early_stopping_handler = EarlyStopping(patience=3,
                                           score_function=score_function,
                                           trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    desc = "ITERATION - loss: {:.2f} - tl acc: {:.2f} - tl recall: {:.2f} "
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0, 0, 0)
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        iteration_metrics = engine.state.metrics
        iteration_loss = engine.state.output[2]  # accordingly to trainer's output transform
        pbar.desc = desc.format(iteration_loss,
                                iteration_metrics['tl_accuracy'],
                                iteration_metrics['tl_recall'])
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        tqdm.write(f"Training Results - Epoch: {engine.state.epoch} - "
                   f"Avg TL accuracy: {metrics['tl_accuracy_avg']:.2f} - "
                   f"Avg Dice: {metrics['seg_dice']}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        tqdm.write(f"Validation Results - Epoch: {engine.state.epoch} - "
                   f"Avg TL accuracy: {metrics['tl_accuracy_avg']:.2f} - "
                   f"Avg Dice: {metrics['seg_dice']}")

        pbar.n = pbar.last_print_n = 0

    wandb_logger = WandBLogger(
        project="tsad",
        entity="autonomous-driving",
        name="ad-encoder",
        config={"max_epochs": args.epochs, "batch_size": args.batch_size},
        tags=["pytorch-ignite", "ad-encoder"]
    )

    model_checkpoint = ModelCheckpoint(
        wandb_logger.run.dir, n_saved=2, filename_prefix='best',
        require_empty=False, score_function=score_function,
        score_name="validation_accuracy",
        global_step_transform=global_step_from_engine(trainer)
    )
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint,  {'model': model})

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        metric_names="all",
        output_transform=lambda loss: {"loss": loss[2]},
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    wandb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names="all",
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    wandb_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names="all",
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    wandb_logger.watch(model)

    trainer.run(train_loader, max_epochs=args.epochs)
    pbar.close()


if __name__ == '__main__':
    import sys
    import torch.optim as optim

    sys.path.append('..')
    sys.path.append('.')
    from models.encoder_loss import ADLoss
    from models.ADEncoder import ADEncoder
    import torch
    from torch import optim
    from models.carlaDatasetSimple import CarlaDatasetSimple
    import argparse

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../dataset', type=str, help='Path to dataset folder')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size.')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of data loader workers')

    # weights
    parser.add_argument('--loss-weights', default="1, 1, 1, 1", type=str,
                        help='Loss weights [segmentation, traffic light status, vehicle affordances ]')
    parser.add_argument('--tl-weights', default="0.2, 0.8", type=str,
                        help='Traffic light weights [Green, Red]')
    parser.add_argument('--pd-weights', default="0.2, 0.8", type=str,
                        help="Pedestrian loss weights [no pedestrian, pedestrian]")

    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate.')
    args = parser.parse_args()

    weights_to_tuple = lambda x: tuple([float(s) for s in str(x).split(",")])
    args.loss_weights = weights_to_tuple(args.loss_weights)
    args.tl_weights = weights_to_tuple(args.tl_weights)
    args.pd_weights = weights_to_tuple(args.pd_weights)

    run(args)
