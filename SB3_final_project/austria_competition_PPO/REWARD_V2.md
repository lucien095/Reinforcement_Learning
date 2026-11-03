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
| overspeed | `-1.5 * max(0, speed - allowed_speed(progress))` |
| steering change | `-0.05 * abs(steer - prev_steer)` |
| stall | `-0.3` per step while speed < 0.3 m/s; after 150 such steps in a row (3 s) the episode ends with `-10` |
| checkpoint | `+0.5` |
| wall crash | `-(5 + 2 * speed)` and the episode ends |
| wrong way | `-5` and the episode ends |

There is no throttle bonus. `allowed_speed` is the corner speed limit (`sqrt(a_lat / curvature)`) lowered by
a braking envelope (`a_brake`), so the penalty starts several meters BEFORE the corner and teaches the agent
to brake early. For the hairpin (progress about 0.51 to 0.55) the limit is about 2.0 m/s.

## Parking exploit (found in the first training run)

With a small stall penalty (`-0.05`) the first policy learned to stop in front of corners: parking costs
almost nothing, while crashing costs 5 to 15. The stall penalty is now large and a long stall ends the
episode, so standing still is always worse than trying to drive. An idle car scores about -55 per episode, a
full-throttle crash about +5.

## Curriculum

With probability `hard_start_prob` (default 0.5) an episode starts 0.06 to 0.16 progress (about 5 to 15 m)
before a sharp corner, so the car has time to accelerate and arrives at the corner fast. Half of those
starts are replayed from places where the environment crashed recently. The remaining episodes use the
original random start.

## Measured vehicle limits

At full steering lock the car turns with a radius of about 1.5 m at 1.6 m/s and about 1.8 m at 2.2 m/s, and the
radius grows with speed. The hairpin centerline radius is about 2 m, so entering it at 3 m/s is physically not
possible. This is why `a_lat` is 2.8 (measured lateral acceleration is about 2.6 m/s^2) and the overspeed
penalty is 1.5 per m/s. A first run with `a_lat=4.0` and a penalty of 0.5 drove into the hairpin exit at
3.1 m/s with almost no steering, in 10 out of 10 test episodes.

## Tuning

- `a_lat` and `a_brake` in `SpeedProfile` set how conservative the envelope is. If the car crawls through
  corners, raise `a_lat`. If it still crashes at the hairpin, lower it.
- Watch `overspeed` and `allowed_speed` in the step `info`, and the episode reward in TensorBoard
  (`./PPO_v2_tensorboard/`).
- At evaluation time use the same wrappers as `PPO_v2.py` (gray scale, resize 84, frame stack 8). The policy
  uses only the camera image, so the competition server needs no change.

## Training log (4 CPU cores, 8 environments, about 70 steps/s)

| Run | Change | Hairpin pass rate (10 episodes) |
| --- | --- | --- |
| 1 | small stall penalty | 0/10, the car parked in front of corners |
| 2 | stall penalty 0.3 and stall termination | 0/10 at 144k steps, drove into the hairpin exit at 3.1 m/s |
| 3 | `a_lat` 2.8, overspeed penalty 1.5, resumed from run 2 | 4/10 at 128k, then 0/10 at 208k (over-braking) |
| 4 | resumed from the 4/10 checkpoint, constant lr 1e-4 | 4/10 at 64k and 144k, then 1/10 at 208k |

`models_v2/ppo_v2_hairpin_4of10.zip` is the best checkpoint (4/10). Evaluate it with
`PYTHONPATH=. python evaluate_hairpin.py models_v2/ppo_v2_hairpin_4of10.zip 10`.

Known problem: in runs 3 and 4 the policy degrades after about 200k steps by braking far too hard in front of
the hairpin (motor -1, speed near 0, ending against the wall). Ideas: reduce the overspeed penalty or cap it
per step, add a reward for keeping speed close to the allowed speed, use more environments, or train longer
with checkpoint selection by hairpin pass rate.
