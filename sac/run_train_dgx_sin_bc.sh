python train_sac.py --host 172.18.0.1 \
  --port 2004 \
  --num-seed 4000  --vehicles 100 --walkers 100 \
  --batch-size 256 --max-episode-steps 300 --eval-frequency 50 \
  --reward-scale 10 --learn-temperature --act-mode pid \
  --num-train-steps 1000000 --num-eval-episodes 3 --online-memory-size 25000 --wandb