python train.py --host 172.18.0.1 --num-seed 4000 --port 2004 --vehicles 100 --walkers 100 \
  --vis-weights '../dataset/models/visual_weights.pt' \
   --temp-weights '../dataset/models/best_SequenceRNNEncoder.pth' \
  --batch-size 512 --max-episode-steps 400 --eval-frequency 50 \
  --bc '../dataset/carla_v7_clean_encodings/carla_v7_clean_encodings' \
  --reward-scale 10 \
  --num-train-steps 1000000 --num-eval-episodes 3 --online-memory-size 16384