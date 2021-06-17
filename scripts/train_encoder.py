from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from termcolor import colored
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from ignite.contrib.handlers.wandb_logger import WandBLogger


def prepare_batch(batch, device, non_blocking):
    y_expected = {}
    for t, label in zip(batch[1:], ["segmentation", "traffic_light_status", "vehicle_affordances", "pedestrian"]):
        y_expected[label] = t.to(device=device, non_blocking=non_blocking)
    x = batch[0].to(device=device, non_blocking=non_blocking)
    return x, y_expected


def trainer_output_transform(x, y, y_pred, loss):
    return {
        'y': y,
        'y_pred': y_pred,
        'loss': loss
    }


def output_transform_tl(output):
    """
    Output transform for traffic light status metrics.
    """
    y_pred = output['y_pred']['traffic_light_status'].argmax(dim=1)
    y = output['y']['traffic_light_status'].argmax(dim=1)
    return y_pred, y    # output format is according to `Accuracy` docs


def wandb_iteration_completed_transform(output):
    return output


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

    tl_accuracy = Accuracy(output_transform=output_transform_tl)
    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        loss,
                                        prepare_batch=prepare_batch,
                                        output_transform=trainer_output_transform,
                                        device=device)
    tl_accuracy.attach(trainer, "tl_accuracy")
    evaluator = create_supervised_evaluator(model, metrics={}, device=device)  # TODO: include required metrics

    desc = "ITERATION - loss: {:.2f} - tl-acc: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0, 0)
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output['loss'],
                                tl_accuracy.compute())
        pbar.update(log_interval)
        # wandb.log({"train loss": engine.state.output})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        # TODO: Put here the desired metrics
        avg_accuracy = 0
        avg_nll = 0
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0
        # wandb.log({"validation loss": avg_nll})
        # wandb.log({"validation accuracy": avg_accuracy})

    '''
    Wandb Object Creation
    '''
    wandb_logger = WandBLogger(
        project="tsad",
        name="ad-encoder",
        config={"max_epochs": args.epochs, "batch_size": args.batch_size},
        tags=["pytorch-ignite", "ad-encoder"]
    )

    '''
    Attach the Object to the output handlers:
    1) Log training loss - attach to trainer
    2) Log validation loss - attach to evaluator
    3) Log optional Parameters
    4) Watch the model
    '''
    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        metric_names=["tl_accuracy"]
    )

    wandb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["nll", "accuracy"],
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    wandb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name='lr'  # optional
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
    parser.add_argument('--pd-weights', default="0.8, 0.2")

    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate.')
    args = parser.parse_args()

    weights_to_tuple = lambda x: tuple([float(s) for s in str(x).split(",")])
    args.loss_weights = weights_to_tuple(args.loss_weights)
    args.tl_weights = weights_to_tuple(args.tl_weights)
    args.pd_weights = weights_to_tuple(args.pd_weights)

    assert len(args.tl_weights) == 2

    run(args)
