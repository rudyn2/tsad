python train.py --host 172.18.0.1 --num-seed 4000 --port 2004 --vehicles 100 --walkers 100 \
  --batch-size 512 --max-episode-steps 400 --eval-frequency 50 \
  --reward-scale 10 --learn-temperature \
  --num-train-steps 1000000 --num-eval-episodes 3 --online-memory-size 25000