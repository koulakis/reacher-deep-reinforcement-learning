# PPO
python scripts/train_agent.py \
  --experiment-name zoo_ppo \
  --agent-type multi \
  --rl-algorithm ppo \
  --n-envs 16 \
  --total-timesteps 2000000 \
  --batch-size 128 \
  --n-steps 512 \
  --gamma 0.99 \
  --gae-gamma 0.99 \
  --n-epochs 20 \
  --learning-rate 0.00003 \
  --clip-range 0.4 \
  --policy-layers-comma_sep "256,256" \
  --value-layers-comma_sep "256,256" \
  --log-std-init -2 \
  --ortho-init