# PPO on Pendulum-v1

Train a continuous-control policy with **PPO (Proximal Policy Optimization)** in Gym’s `Pendulum-v1` environment. The implementation uses an Actor–Critic architecture, GAE(λ), clipped surrogate objectives, and advantage normalization. The training script saves the best Actor weights under `models/` and logs episode returns with TensorBoard.

## Requirements

- Python 3.9+ (3.10 recommended)
- Windows / Linux / macOS  
  GPU is optional: `torch` uses CUDA automatically when a CUDA build of PyTorch is installed.

## Installation

```bash
pip install -r requirements.txt
```

To install a CUDA-enabled PyTorch build for your platform, follow the [official PyTorch install guide](https://pytorch.org/get-started/locally/) first, then install the rest of this project’s dependencies (you can temporarily remove the `torch` line from `requirements.txt` so it is not overwritten by a CPU-only wheel).

## Project layout

| File | Description |
|------|-------------|
| `ppo_agent.py` | Actor / Critic, `ReplayMemory`, `PPOAgent` (rollouts, GAE, multi-epoch updates, policy checkpointing) |
| `ppo_training.py` | Training loop: `Pendulum-v1`, hyperparameters, logging, and model saves |
| `ppo_test.py` | Loads a trained Actor and runs a visual rollout (`pygame` + Classic Control rendering) |

Training produces:

- `models/ppo_actor_fixed_<timestamp>.pth` — best Actor weights so far  
- `fixed_logs/` — TensorBoard event files  
- `fixed_reward_<timestamp>.txt` — per-episode return array  

## Training

From the repository root:

```bash
python ppo_training.py
```

View scalars:

```bash
tensorboard --logdir=fixed_logs
```

## Evaluation / demo

After training, ensure `models/` contains at least one `ppo_actor_fixed_*.pth`, then run:

```bash
python ppo_test.py
```

By default the script loads the **last** `ppo_actor_fixed_*.pth` when sorted by filename under `models/`. Press **Esc** or close the window to quit.

## License

Use this repository in line with your course or organization’s terms if this code comes from educational materials.
