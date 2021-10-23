import argparse
from bc_agent import BCStochasticAgent
from models.carlaAffordancesDataset import AffordancesDataset
import matplotlib.pyplot as plt

HLC_TO_NUMBER = {
    'RIGHT': 0,
    'LEFT': 1,
    'STRAIGHT': 2,
    'LANEFOLLOW': 3
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot single episode predictions",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to data folder')
    parser.add_argument('--act-mode', default="raw", type=str, help="Action space. 'raw': raw actions (throttle, brake,"
                                                                    "steer), 'pid': (target_speed, steer)")
    parser.add_argument('--eval-frequency', default=50, type=int)
    parser.add_argument('--checkpoint', default='checkpoint.pt', type=str)
    parser.add_argument('--next-speed', action='store_true', help='Whether to use next step speed head or not.')
    parser.add_argument('--norm-actions', action='store_true', help='Whether the action are normalized or not.')
    args = parser.parse_args()

    # create agent and load the parameters
    action_dim = 2 if args.act_mode == "pid" else 3
    agent = BCStochasticAgent(input_size=15, hidden_dim=512,
                              action_dim=action_dim, log_std_bounds=(-2, 5),
                              checkpoint=args.checkpoint, next_speed=args.next_speed)
    agent.load()
    agent.eval_mode()

    # load one episode
    dataset = AffordancesDataset(args.data)

    ep_aff, ep_ctrl, ep_speed, ep_cmd = dataset.get_episode(source="val", normalize_control=True)
    target_speed = []
    real_steer = []
    pred_steer = []

    # evaluate
    for aff, ctrl, speed, cmd in zip(ep_aff, ep_ctrl, ep_speed, ep_cmd):
        task = HLC_TO_NUMBER[cmd]
        action = agent.act_single(obs=dict(affordances=aff), task=task)     # speed, steering
        real_steer.append(ctrl[2])
        target_speed.append(action[0])
        pred_steer.append(action[1])

    # plot results
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

    # plot pred speed vs target speed
    axs[0].plot(target_speed, label="pred speed")
    axs[0].plot(ep_speed, label="real speed")
    axs[0].set_ylabel("Normalized speed")
    axs[0].set_xlabel("Step")
    axs[0].legend()

    # plot pred steer vs real steer
    axs[1].plot(pred_steer, label="pred steer")
    axs[1].plot(real_steer, label="real steer")
    axs[1].set_ylabel("Normalized steering")
    axs[1].set_xlabel("Step")
    axs[1].legend()

    plt.tight_layout()
    plt.show()



