# PPO for Pendulum-v1 control problem: test loop
# Born time: 2024-09-01
# Dylan

import os
import glob
import numpy as np
import torch
import pygame

# --------------------- pygame / gym compatibility shims ----------------------
# Two issues with gym 0.26.2 + pygame 2.6 are addressed below:
#   1. pygame.transform.smoothscale rejects numpy.float32 sizes
#      ("size must be two numbers"). We coerce to int.
#   2. pendulum.py reloads "clockwise.png" from disk every single frame, which
#      tanks the framerate. We cache the loaded surface so each unique path
#      hits the disk only once.

_orig_smoothscale = pygame.transform.smoothscale


def _safe_smoothscale(surface, size, *args, **kwargs):
    size = (int(size[0]), int(size[1]))
    return _orig_smoothscale(surface, size, *args, **kwargs)


pygame.transform.smoothscale = _safe_smoothscale

_image_cache: dict = {}
_orig_image_load = pygame.image.load


def _cached_image_load(path, *args, **kwargs):
    key = os.fspath(path) if hasattr(os, "fspath") else str(path)
    surf = _image_cache.get(key)
    if surf is None:
        surf = _orig_image_load(path, *args, **kwargs)
        _image_cache[key] = surf
    return surf


pygame.image.load = _cached_image_load
# -----------------------------------------------------------------------------

import gym  # noqa: E402
from ppo_agent import Actor  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Computing device:", device)

# Use gym's own window. render_mode="human" makes env.step() render
# automatically, so we MUST NOT call env.render() ourselves (it would double
# every frame and halve the visual framerate).
env = gym.make("Pendulum-v1", render_mode="human")

# Pendulum's default render_fps is 30, which feels choppy. Bump it.
TARGET_FPS = 60
env.unwrapped.metadata["render_fps"] = TARGET_FPS

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]


# -------------------------- Locate checkpoint --------------------------------
current_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_path, "models")

# Hard-code a specific checkpoint by setting actor_path. Leave it as None to
# auto-pick the most recent ppo_actor_fixed_*.pth in the models/ folder.
actor_path = None

if actor_path is None:
    candidates = sorted(glob.glob(os.path.join(model_dir, "ppo_actor_fixed_*.pth")))
    if not candidates:
        raise FileNotFoundError(
            f"No ppo_actor_fixed_*.pth found in {model_dir}. "
            "Train with ppo_training.py first."
        )
    actor_path = candidates[-1]

print(f"Loading actor from: {actor_path}")

actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path, map_location=device))
actor.eval()


@torch.no_grad()
def deterministic_action(s):
    """At evaluation time we take the mean of the policy (no exploration)."""
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    mean, _ = actor(s_t)
    return mean.clamp(-2.0, 2.0).cpu().numpy()[0]


# -------------------------- Rollout ------------------------------------------
num_episodes = 10
episode_rewards = []
stopped = False

try:
    for episode in range(num_episodes):
        if stopped:
            break
        state, _ = env.reset(seed=episode)
        episode_reward = 0.0

        for step_i in range(200):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    stopped = True
                    break
            if stopped:
                break

            action = deterministic_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            # NOTE: do not call env.render() here — see comment above env creation.

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    env.close()
    if episode_rewards:
        print(
            f"Summary over {len(episode_rewards)} episodes: "
            f"mean={np.mean(episode_rewards):.2f}, "
            f"max={np.max(episode_rewards):.2f}, "
            f"min={np.min(episode_rewards):.2f}"
        )
