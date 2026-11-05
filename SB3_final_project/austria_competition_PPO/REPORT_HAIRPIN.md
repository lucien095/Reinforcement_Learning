# Why the car could not pass the hairpin, and what finally worked

This report is a field guide for future RL projects. It covers the Austria racecar_gym competition (camera only,
continuous motor and steering, 100 s per run, score = laps + progress). It explains why the original PPO runs never
got through the hairpin, every bottleneck met while fixing it, the solution that worked, and reusable rules for
reward design and for the surrounding engineering.

## 1. Summary

| Question | Answer |
| --- | --- |
| Did pure PPO ever pass the hairpin reliably? | No. The best RL checkpoint passed 4 out of 10 hairpin tests and degraded when training continued. |
| What finally got the car around the track? | Behavior cloning of a scripted expert that uses privileged simulator state, then DAgger (the camera policy drives, the expert labels). |
| Final score in the staff competition environment | Mean 1.53 over 24 runs of 100 s (13 runs finished about 2 laps without a crash, scores 1.9 to 2.0; 5 runs scored below 1.0 because of a crash or a stall). The best 8-run sample read 1.92, which was partly luck. |
| Final model | `models_v2/bc_dagger_iter8.zip`, run with `agent_v2.py`. |

The honest limits: the result is not reliable on every run (see section 6), and the pure RL part of the project
did not solve the task. The expert plus imitation route solved most of it.

## 2. Why the hairpin was not passed (root causes, with evidence)

These are ordered by how much each one mattered.

### 2.1 The reward paid for full speed and made crashing cheap

The original reward per step was `motor_action + 1000 * progress - 0.1 * |steer change| + checkpoint bonus - small wall penalty`.

* `motor_action` pays for throttle everywhere, also in front of a corner.
* Crashing ended the episode but cost only 2 to 5 points (later experiments used 100).
* Result: full throttle into the corner is a stable local optimum for PPO. Passing the hairpin needs a precise
  sequence (brake early, steer to the lock, accelerate), which random exploration finds only rarely.
* Evidence: in all failing evaluations the car reached the corner at 3.1 to 3.3 m/s with the motor at +1.0.

### 2.2 The physics: the hairpin cannot be taken at the car's top speed

Measured in the simulator:

* Top speed is about 3.3 m/s (motor command 1.0).
* At full steering lock the minimum turning radius is about 1.5 m at 1.6 m/s and about 1.8 m at 2.2 m/s, and it
  grows with speed. The measured lateral acceleration limit is about 2.6 m/s^2.
* The hairpin center line has a radius of about 2 m, so the safe speed is about 2 m/s or less.

Nothing in the first reward told the agent about a speed limit that depends on the place on the track.

### 2.3 The braking delay made "braking" look the same as "stopping"

Braking has a delay, so frames with `motor > 0` and `progress == 0` happen after braking. The team notes show the
consequence: whatever penalty was tuned, the agent learned to brake randomly or to stop. Adding stop penalties made
it brake more, ending the episode on stops made the data too clean to learn from, and removing the penalties
brought the random braking back. This was an unstable loop of reward tweaks, not a fix.

### 2.4 Perception: the policy cannot see the quantities the reward uses

The policy sees only a 128x128 first person camera (converted to 84x84 gray, 8 stacked frames). Wall distance, speed
and position on the track are not inputs. The wall is the only distance cue, and it grows in the image late. The
trained car steered almost straight until the wall was 0.3 away and then steered to the lock, too late.
Colors do not help: sky, walls and floor are all low saturation gray tones.

### 2.5 The sample budget was far below what the reference solution used

A classmate's report (same task) solved the hairpin with the same plain reward, PPO, random start positions, 80
parallel environments and 2 to 5 million steps, and states that identical settings gave very different results
between seeds. The experiments in this project ran 0.1 to 0.3 million steps per attempt on 4 CPU cores (about 70 to
140 steps per second). Those runs were too short to say that a reward idea failed, and the evaluation of 10 episodes
was too noisy to rank them.

### 2.6 The training environment was not the evaluation environment

This one was discovered late and changed the interpretation of every earlier result.

| Property | Training wrapper (`racecar_gym/env.py`, modified) | Staff environment used by the server (`Demo/racecar_gym/env.py`) |
| --- | --- | --- |
| Start position | random | the start line (grid) |
| Crash | episode ends | the car is put back on the track and `30 + 10 * speed^2` seconds are removed from the 100 s |
| `wrong_way` | episode ends | ignored |
| Lap counter | broken (see below) | works |

* In the modified wrapper `info["checkpoint"] = 4` is written when the state reports checkpoint 5. The info dictionary is
  the live state dictionary, so the lap detection (which needs checkpoint 5) never fires. After the first lap the
  wrong-way check triggers at progress 0.6 and ends the run. Any scoring done with that wrapper stops at 0.6.
* A crash above about 2.6 m/s costs more than the whole time limit, so a single fast crash ends a staff run.
* `evaluate_competition.py` now uses an exact copy of the staff environment (`racecar_gym/env_competition.py`).

### 2.7 The test only covered one corner

The hairpin tests started before progress 0.507 only. A second sharp corner at progress 0.32 to 0.37 (about 136
degrees) was where most staff-environment runs actually ended. Measuring the real metric (the score in the real
environment) found this after many hours.

### 2.8 Smaller issues that made things worse

* Parking exploit: with a small stall penalty the policy stopped in front of corners (no crash, tiny cost).
* Over-braking: a strong overspeed penalty (1.5 per m/s) produced policies that braked to almost zero and pressed
  against the wall. Reducing it to 0.8 with a cap fixed the over-braking but exposed the late steering of 2.4.
* RL training degraded after about 200k steps in several runs (reward oscillation), so the last checkpoint was often
  not the best.

## 3. Timeline of attempts

| Step | Change | Result |
| --- | --- | --- |
| 1 | Analysis of the original code and notes | Root causes 2.1 to 2.3 identified. |
| 2 | Reward v2: progress per meter, overspeed penalty from a braking envelope, stall penalty, speed-scaled crash penalty, hairpin curriculum starts | The first policy parked in front of corners. |
| 3 | Stall penalty 0.3 per step and episode end after 3 s of standing still | No more parking, still 0/10 at the hairpin (drove in at 3.1 m/s). |
| 4 | Measured the turning radius; `a_lat` 4.0 -> 2.8, overspeed penalty 0.5 -> 1.5 | 4/10 at the hairpin at best, then degraded (over-braking). |
| 5 | Lower learning rate (1e-4), overspeed penalty 0.8 with a cap, near-wall penalty | 0/10 to 2/10, steering was too late. |
| 6 | Read the classmate's report | Same reward, different scale of training. Random starts and long runs matter. |
| 7 | Found that evaluation used the wrong environment and one corner only | New evaluator based on the staff environment. |
| 8 | Scripted expert with privileged state (pure pursuit + braking envelope), tuned with a grid search | 8/8 passes at both sharp corners and about 2 laps in 100 s without a crash (score 1.99). |
| 9 | Behavior cloning of the expert into the CNN policy | 0.3 score: the policy got stuck in states the expert never showed. |
| 10 | DAgger (the policy drives, the expert labels every state) | Iteration 5: 1.80 on 8 runs; iteration 8: 1.92 on 8 runs, 1.53 on 24 runs. |
| 11 | Faster expert (42.6 s per lap) | The CNN could not copy the tighter braking: score 0.6 to 1.1. Rejected. |
| 12 | Model averaging of iterations 6 to 9; scaling the motor output | Worse (1.25 and 0.95). Rejected. |

## 4. What worked and why

### 4.1 A scripted expert that uses privileged information

The expert (`expert_driver.py`) reads the true pose, speed and progress. Steering is pure pursuit toward a look
ahead point on the track center line. The speed target is the lowest allowed speed over the next 3 m, where the
allowed speed comes from the corner curvature (`sqrt(a_lat / curvature)`) and a braking envelope computed backwards
around the lap (`speed_profile.py`). The center line and the curvature are computed offline from the track maps.
It needs no training. Its parameters were found by a grid search over 24 settings (8 test runs per setting).

Why it matters for RL work: the expert is a baseline, a sanity check of the environment (it showed the lap counter
bug), a measurement tool (the best possible lap time) and a teacher at the same time.

### 4.2 Behavior cloning with DAgger

1. Iteration 0: the expert drives with noise bursts (random steering and throttle offsets held for 8 to 25 steps)
   so the data also contains off-line states. Every frame is labeled with the expert action.
2. Train the same CNN policy used for PPO (8 stacked 84x84 gray frames) with a mean squared error on the actions
   (steering weighted twice).
3. DAgger iterations: the current policy drives, the expert labels the states that the policy reaches, the data is
   added and the policy is retrained. This is what fixed the stuck and crawling states of iteration 0.
4. Start positions are biased toward the two sharpest corners.

### 4.3 Evaluate with the real metric

All decisions after step 7 used the staff environment, the real score, and 24 runs for the final choice.

## 5. Reward design guide (reusable)

### 5.1 Terms that worked in this project

| Term | Form | Role |
| --- | --- | --- |
| Progress | `k * meters driven` (about 10 per m) | main signal, measured in physical units, scale of about 0.5 to 1 per step at top speed |
| Overspeed | `-k * min(cap, max(0, speed - allowed_speed(place)))` | teaches braking before a corner; the allowed speed includes a braking envelope so the penalty starts early |
| Stall | `-0.3` per step below 0.3 m/s, episode end after 150 stalled steps | removes the parking exploit; must be worse than any attempt |
| Crash | `-(5 + 2 * speed)` and episode end | a crash at high speed costs more |
| Wall proximity | `-1.0 * max(0, (0.5 - wall_dist) / 0.5)` | dense warning; zero on the center line, so it does not bias the line |
| Steering change | `-0.05 * |delta steer|` | smoothness |
| Checkpoint | small bonus | optional |

### 5.2 Rules of thumb

1. Put the reward in physical units (meters, m/s) so the scale is easy to reason about. The raw environment reward
   (progress per step below 0.0016) is too small for a value network.
2. Do not reward the raw action (throttle). Reward the result (distance) and penalize the unsafe state (speed above
   the limit for this place).
3. Every exploit needs a closed form: if standing still is cheap, the agent will stand still. Make sure
   "do nothing" is the worst policy, by testing an idle agent and a full-throttle agent in a script and
   comparing their returns (here -55 and +5 per episode).
4. A penalty that is too strong creates a new exploit (over-braking). Cap the penalty per step.
5. Penalties for a state need a way for the agent to perceive that state. Check what the policy can actually see.
6. Termination changes what is learned. Early termination on a bad event is a strong signal, but ending episodes
   for mild events removes data.

## 6. Technique checklist for other RL projects

1. Verify the training environment against the evaluation environment before anything else: start position, reset
   rules, termination, time limit, noise, scoring formula, wrappers. Diff the staff code against your own.
2. Use the real metric for model selection, with enough episodes (at least 24 here; 8 gave 1.92 and 24 gave 1.53 for
   the same model).
3. Build a scripted expert first when the simulator exposes privileged state. Measure the physical limits (top
   speed, turning radius, braking) with small scripts instead of guessing parameters.
4. Visualize what the policy sees (a saved image takes one minute) and decide the preprocessing from it.
5. Diagnose failures with the last 10 to 15 steps before each failure: observation, policy action, expert action,
   speed. In this project the late steering and the full-throttle approach were visible immediately.
6. Use curriculum starts (before the hardest places) and random starts so rare events are seen often.
7. Imitation first, RL second: behavior cloning plus DAgger gives a competent initial policy; RL fine-tuning can
   follow from it (it was not needed here, and it was not tried).
8. Keep checkpoints and pick by evaluation score. Training curves can degrade after the peak.
9. Match the compute to the claim: a reward idea tested for 0.2 million steps on 8 environments is not tested.
10. Log every experiment with its result (see the table in section 3), including the failures.
11. Engineering notes from this session: `pkill -f` and `pgrep -f` can match the shell running the command and kill
    it, so use the bracket trick (`[p]attern`) or kill by process id; large frame datasets should store single
    frames and rebuild the stacks per batch (600k frames of 84x84 are 4 GB, 8x larger when stacked).

## 7. Known limitations of the delivered model

* Mean 1.53 over 24 runs, individual runs between 0.03 and 2.0. Failures are crashes at the corner around progress
  0.36 (the car arrives at 1.5 to 1.7 m/s, the expert at about 1.2 m/s, the policy averages the expert braking
  with non-braking states) and occasional crawling or stalling after a slow state.
* Quality varies between DAgger iterations (24 runs each: iteration 5 1.53, iteration 6 1.08, iteration 7 1.48,
  iteration 8 1.53), so more iterations are not automatically better; pick by evaluation.
  `models_v2/bc_dagger_iter5.zip` and `models_v2/bc_dagger_iter8.zip` are equally good.
* The expert speed was limited to what the CNN could imitate (50 s per lap). The theoretical top is about 28 s per
  lap.

## 8. Ideas to push further

1. Make the expert slower and more conservative only at the two sharp corners (a safety margin for imitation
   error) and rerun DAgger.
2. Train longer with a lower learning rate on all collected data, and weight the DAgger data of the failure states
   more.
3. Give the network a better view: crop the floor (about half of the image carries no information), use a higher
   resolution, or add a recurrent layer. The classmate noted that a single frame does not tell whether the hairpin
   turns left or right, which is the reason to use stacked frames or memory.
4. Fine-tune the cloned policy with PPO at a low learning rate and a small entropy bonus, using the staff
   environment for evaluation, and keep the best checkpoint.
5. Run on a machine with many more cores: the reference solution used 80 environments for several million steps.

## 9. Reproduce

```
cd SB3_final_project/austria_competition_PPO
# score a model like the competition server does
PYTHONPATH=. python evaluate_competition.py models_v2/bc_dagger_iter8.zip 4
# use it in the competition (server.py running on port 5000)
python agent_v2.py --model models_v2/bc_dagger_iter8.zip --url http://localhost:5000
# rebuild the pipeline
PYTHONPATH=. python bc_collect.py --out-dir data/iter0 --steps 30000 --workers 4
PYTHONPATH=. python bc_train.py --data data/iter0 --init models_v2/ppo_v2_hairpin_4of10.zip --out bc_iter0.zip --epochs 6
PYTHONPATH=. python bc_collect.py --out-dir data/iter1 --steps 20000 --workers 4 --policy bc_iter0.zip --beta 0.2
PYTHONPATH=. python bc_train.py --data data/iter0 data/iter1 --init bc_iter0.zip --out bc_iter1.zip --epochs 4
```

The code needs Python 3.10 or older (the old dataclass usage in the environment package fails on 3.11 and newer),
`yamldataclassconfig==1.5.0` and the packages from `requirements.txt`.

Files: `expert_driver.py`, `racecar_gym/speed_profile.py`, `bc_collect.py`, `bc_train.py`, `agent_v2.py`,
`evaluate_competition.py`, `racecar_gym/env_competition.py`, `racecar_gym/env_v2.py` (reward v2),
`PPO_v2.py` (RL training), `REWARD_V2.md` (reward details and the RL training log).
