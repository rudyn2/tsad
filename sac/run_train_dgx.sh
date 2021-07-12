python train.py --host 172.18.0.1 --num-seed 4000 --port 2004 --vehicles 100 --walkers 0 \
  --vis-weights '../dataset/weights/best_model_11_validation_accuracy=-1.6713.pt' \
   --temp-weights '../dataset/weights/best_VanillaRNNEncoder(2).pth' \
  --batch-size 512 --max-episode-steps 400 --eval-frequency 12 \
  --num-train-steps 1000000 --num-eval-episodes 3 --online-memory-size 16384