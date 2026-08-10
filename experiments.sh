python train.py \
  --output-dir artifacts/1B_l9_w64 \
  --train-tokens 1000000000 \
  --validation-tokens 100000000 \
  --checkpoint-every 250000000 \
  --activation-layer 9 \
  --width-multiplier 64 \
  --k 16 --resume

python record_activations.py --sae-dir artifacts/1B_l9_w64

python interpret_features.py \
  --analysis artifacts/1B_l9_w64/analysis \
  --base-url http://127.0.0.1:9000/v1 \
  --api-key llamacpp \
  --model model_default \
  --no-reasoning \
  --concurrent 4