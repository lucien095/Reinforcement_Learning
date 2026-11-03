# Reward V2 (hairpin fix)

Files:
- `racecar_gym/speed_profile.py`: builds a curvature based braking envelope from the track maps (numpy only).
  Run `python racecar_gym/speed_profile.py` to print the corner entries and the allowed speed per progress.
- `racecar_gym/env_v2.py`: `RaceEnvV2`, same observation/action spaces as `RaceEnv`.
- `PPO_v2.py`: training script (`python PPO_v2.py --env-num 50`, `--resume <checkpoint.zip>` to continue).
- `racecar_gym/envs/gym_api/single_agent_race.py`: `reset(options={'start_pose': ...})` is now supported.

## Why the old reward failed at the hairpin

- `r_motor = motor_action` paid for full throttle everywhere. Crashing only cost -2 and ended the episode,
  so driving at full speed into the hairpin was a stable local optimum.
- Braking has a delay, so `motor > 0` with `progress == 0` happens after braking. The agent learned that
  stopping or braking randomly is acceptable, and every penalty tweak only moved the problem around.

## New reward (per step)

| Term | Value |
| --- | --- |
| progress | `+10 * meters driven forward` (negative when driving backwards) |
| overspeed | `-0.5 * max(0, speed - allowed_speed(progress))` |
| steering change | `-0.05 * abs(steer - prev_steer)` |
| stall | `-0.05` when speed < 0.2 m/s where the allowed speed is above 1.5 m/s |
| checkpoint | `+0.5` |
| wall crash | `-(5 + 2 * speed)` and the episode ends |
| wrong way | `-5` and the episode ends |

There is no throttle bonus. `allowed_speed` is the corner speed limit (`sqrt(a_lat / curvature)`) lowered by
a braking envelope (`a_brake`), so the penalty starts several meters BEFORE the corner and teaches the agent
to brake early. For the hairpin (progress about 0.51 to 0.55) the limit is about 2.4 m/s.

## Curriculum

With probability `hard_start_prob` (default 0.5) an episode starts 0.06 to 0.16 progress (about 5 to 15 m)
before a sharp corner, so the car has time to accelerate and arrives at the corner fast. Half of those
starts are replayed from places where the environment crashed recently. The remaining episodes use the
original random start.

## Tuning

- `a_lat` and `a_brake` in `SpeedProfile` set how conservative the envelope is. If the car crawls through
  corners, raise `a_lat`. If it still crashes at the hairpin, lower it.
- Watch `overspeed` and `allowed_speed` in the step `info`, and the episode reward in TensorBoard
  (`./PPO_v2_tensorboard/`).
- At evaluation time use the same wrappers as `PPO_v2.py` (gray scale, resize 84, frame stack 8). The policy
  uses only the camera image, so the competition server needs no change.
