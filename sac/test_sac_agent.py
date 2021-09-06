from replay_buffer import MixedReplayBuffer


if __name__ == '__main__':
    from sac.agent.sac import SACAgent, DiagGaussianActor, DoubleQCritic
    mixed_replay_buffer = MixedReplayBuffer(512,
                                            reward_weights=(0.3, 0.3, 0.3),
                                            offline_buffer_hdf5='../dataset/encodings/encodings.hdf5',
                                            offline_buffer_json='../dataset/encodings/carla_v7_clean_encodings.json')
    actor = DiagGaussianActor(action_dim=2,
                              hidden_dim=256,
                              log_std_bounds=(-3, 3)
                              )
    critic = DoubleQCritic(action_dim=3,
                           hidden_dim=256)
    target_critic = DoubleQCritic(action_dim=3,
                                  hidden_dim=256)
    agent = SACAgent(actor=actor,
                     critic=critic,
                     target_critic=target_critic,
                     action_dim=2,
                     batch_size=16,
                     offline_proportion=0.25)
    agent.update(mixed_replay_buffer, step=0)