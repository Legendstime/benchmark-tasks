# Place five SMD LEDs on a circuit board with their cathode corners at the marks

A green 100 x 40 mm circuit board sits in a holder between the arms with five 5050 LED footprints along its centreline
at the 19.05 mm key pitch of a five-key macropad: pasted pads, a white silkscreen square, and a white bar across the
corner nearest the robot on its right-hand side, which marks the cathode. Five white 5050 LEDs (5 x 5 x 1.6 mm, a
yellow window on top, one dark cut corner) lie window up on two antistatic trays, turned 60, -45, 30, -60 and 45 degrees
from the correct orientation. Straight flat-tip tweezers stand in a cup near each arm.

Surface-mount parts are placed by hand for prototypes, small batches and rework (O*NET 51-2022.00 Electrical and
Electronic Equipment Assemblers, 51-2028.00 Electrical, Electronic, and Electromechanical Assemblers). The part is too
small for a parallel gripper, there is no socket or hole to guide it, and a 5050 LED has one cut corner: a reversed
LED is dead. The task tests precision placement without mechanical registration (within 1 mm and 10 degrees), reading
a polarity mark, turning the part to match, and tool use with tweezers, on a part a sixth the size of the gripper pads.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()` and the scripted demo |
| `labkit.py` | Parts built from several shapes, with sliding and hinged joints, for yam_arena task files (hooks `yam_arena.scene` from the task's import; yam_arena itself is unchanged) |
| `arena_assets_fix.py` | Imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket (unused here: every object is built in `task.py`) |
| `test_task.py` | Scoring, scene and demo checks without the simulator (`python -m pytest -q test_task.py`) |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from
this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Using the tweezers, place each of the five LEDs on a free footprint on the circuit board, flat on its pads, with its cut corner at the white corner mark. Then stand the tweezers back in their cups." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/smd-led-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Using the tweezers, place each of the five LEDs on a free footprint on the circuit board, flat on its pads, with its cut corner at the white corner mark. Then stand the tweezers back in their cups." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/smd-led-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/smd-led-dream/*.json; do inspect-robots video "$log"; done
```

`task_score` prints the trial's score: 1.0 on success (held 1 s), otherwise `score()` at the end of the episode.

Demo variants, for checking the scorer (each falls short of full score by design):

```bash
DEMO_VARIANT=hold     # places all five but keeps the tweezers out: 0.8
DEMO_VARIANT=norotate # places the parts without turning them: every cut corner is off, 0.0
```

## Scripted demo

Each hand takes its tweezers out of their cup and holds them hanging straight down. For each of its
LEDs it centres the open blades over the part, pinches it by two opposite sides, lifts it, turns it by the angle its
cut corner is off (at most 60 degrees), carries it over its footprint, lines it up watching the part against the
silkscreen, sets it on the pads, nudges it onto the footprint if it has slid, and lets go. The left hand places the three LEDs on the left tray on the
three left-hand footprints, the right hand its two on the right-hand footprints. The hands take turns over the board,
one placement at a time (a gripper is wider along its closing axis than the footprints' spacing); the other hand picks
its next LED meanwhile and holds it until its turn. Both then stand their tweezers back in their cups and move clear.
The demo reads only what the cameras see: the tweezers' tips (to trim each hand's aim while it holds still) and the
LEDs' poses.

The demo's wrist headings can be tuned without editing the file: `TW_HEAD_L`, `TW_HEAD_R` (degrees, the direction
of each hand's preferred jaw axis; -90 by default, the gripper's default grip, so every placement happens there and
the picks stay within 60 degrees of it).

## Scoring

Evaluated every control step; the score at the end of the episode counts.

- 0 if a gripper pad ever came within 1.5 mm of an LED's body (the LED touched by the gripper itself; latched for the
  trial), or if any LED moved more than 1 mm over the last 20 steps (1 s).
- An LED counts as placed on a footprint when its centre is within 1 mm of the footprint's centre (in the board's
  plane), it lies within 0.4 mm of the pads' height, tilts at most 5 degrees, and its cut corner points within
  10 degrees of the footprint's corner bar (towards -x, -y). Each LED and each footprint counts once (nearest first).
- 0.2 per placed LED. 1.0 (success, which must hold 1 s) only when all five are placed, both pairs of tweezers stand
  in their cups (within 10 mm and 15 degrees of their starting pose and 4 mm of its height), both hands have let go
  of them (every pad tip at least 3 cm from either pair's centre line, from between the blades' tips to the top of
  the arms) and both hands' pad tips are at least 8 cm from the board's centre; otherwise at most 0.8.
- An LED upside down (tilt 180), on the wrong side of the outline, off the board, or on top of another LED earns
  nothing; one placed with its corner 90 or 180 degrees off earns nothing.

On a real table: a top photo of each footprint against the silkscreen (the body inside the square to about 1 mm; the
cut corner at the bar), a 0.5 mm ruler or a calliper against the outline for the position, and the video for the rules.

## Runtime and seeds

- The scripted demo finishes with success at step 1,738 (87 s); a full trial is 2,400 steps (2 min at 20 Hz). On an
  L40S the demo run takes about 6 minutes once Isaac Sim is warm (its first start on a fresh instance takes about
  6 minutes more).
- One fixed layout: `-E object_jitter=0`; nothing else is random.

## Fidelity

The LEDs are rigid 5 x 5 x 1.6 mm boxes of 0.3 g (a real 5050 weighs about 0.06 g; the heavier part keeps the contact
solver stable), with a drawn window and cut-corner mark. Pads, solder paste, silkscreen and switch holes are drawn only:
a placed LED rests on the bare board and nothing holds it, so paste tack is not simulated. The tweezers are two rigid
halves on a sprung slide (600 N/m), blades 8 mm wide and 1.2 mm thick, squeezed by the gripper pads. The board, holder,
trays and cups are fixed. Friction 1.1 on the LEDs and tweezers, 0.9 on the trays. Contact offsets 0.5 mm. No
contact-force cap is enforced.

## Real-world build

A 100 x 40 x 1.6 mm FR-4 board with five 5050 (PLCC-6) footprints at 19.05 mm pitch and a cathode bar on each
footprint's silkscreen (any PCB service; a 4 mm hole 5 mm behind each footprint makes it a five-key macropad LED board),
held in a small PCB holder or vise 2 cm above the bench; solder paste on the pads, or tacky flux for repeated trials.
Five 5050 LEDs (for example white 5050 PLCC-6 parts) on two black antistatic trays with printed 6 mm squares and an
arrow per square. Two pairs of straight flat-tip stainless tweezers in cups.
