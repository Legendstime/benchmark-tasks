"""On the station table: a green circuit board (100 x 40 mm) held in a black holder between the arms, with five 5050 LED
footprints along its centreline, each a set of six pasted pads inside a white silkscreen square with a white bar across
the corner nearest the robot on its right-hand side (the cathode corner). Five white 5050 LEDs (5 x 5 x 1.6 mm, a
yellow window on top, one dark cut corner marking the cathode) lie window up on two black antistatic trays, turned at
various angles: three on the left tray, two on the right. Two pairs of straight flat-tip tweezers stand tips down in grey
cups, one near each arm.

The task: with the tweezers, place every LED flat on a free footprint, its cut corner at the footprint's corner bar,
then stand the tweezers back in their cups. LEDs must not be touched by the gripper pads themselves.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- colours (sRGB) ---------------------------------------------------------------------------------------------------
FR4 = (0.05, 0.36, 0.20)  # green solder mask
SILK = (0.95, 0.95, 0.92)  # white silkscreen
PAD = (0.58, 0.59, 0.61)  # pasted pads: grey solder paste
LED_BODY = (0.93, 0.93, 0.90)
PHOSPHOR = (0.96, 0.80, 0.22)  # the yellow window on top of a white LED
MARK = (0.12, 0.12, 0.14)  # the cut cathode corner
ESD_BLACK = (0.07, 0.07, 0.08)
ABS_BLACK = (0.10, 0.10, 0.11)
ALU = (0.70, 0.71, 0.73)
STEEL = (0.72, 0.73, 0.75)
CUP_C = (0.55, 0.56, 0.58)
HOLE = (0.14, 0.14, 0.15)


def _box_mesh(c, size, colour, yaw=0.0):
    """A box drawn only (no collision), optionally turned ``yaw`` about the vertical."""
    c, h = np.asarray(c, float), np.asarray(size, float) / 2
    corners = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    pts = corners @ rot.T + c
    tris = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                     (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return labkit.Mesh(points=pts, tris=tris, color=colour, collide="none")


def _disc_mesh(c, r, colour, n=24):
    """A flat disc drawn only."""
    c = np.asarray(c, float)
    pts = [c] + [c + [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0] for i in range(n)]
    tris = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return labkit.Mesh(points=np.array(pts), tris=np.array(tris), color=colour, collide="none")


# -- the board in its holder (fixed). Frame: origin on the table under the board's centre. ----------------------------
B_XY = (0.30, 0.0)
B_WID, B_LEN, B_T = 0.040, 0.100, 0.0016  # across (x), along (y), thickness
BASE_T = 0.005
LEDGE_Z = 0.020  # the board's underside this far above the table
B_TOP = TOP + LEDGE_Z + B_T  # world z of the board's top face
RAIL_T, RAIL_H = 0.006, 0.028
PITCH = 0.01905
FP_Y = [2 * PITCH, PITCH, 0.0, -PITCH, -2 * PITCH]  # footprints along the board, left to right
FP_YAW = 0.0  # every footprint's cathode corner points towards (-x, -y): nearest the robot, on its right
CORNER_DIR = math.atan2(-1.0, -1.0)  # the LED frame's cut corner, seen from above (rad), at yaw 0
LED_W, LED_H = 0.005, 0.0016  # a 5050 package: 5 x 5 mm, 1.6 mm tall
LED_KG = 0.0003  # 0.3 g in sim (a real 5050 weighs about 0.06 g)
MARK_W = 0.0013
_z = LEDGE_Z + B_T  # the board's top face in the holder frame


def _footprint(y):
    out = []
    # six pasted pads: three along each x side of the footprint
    for sx in (-1, 1):
        for k in (-1, 0, 1):
            out.append(_box_mesh((sx * 0.0024, y + k * 0.0016, _z + 0.00006), (0.0016, 0.0012, 0.0001), PAD))
    # the silkscreen outline, 5.6 mm square, and the corner bar at the (-x, -y) corner
    for sx in (-1, 1):
        out.append(_box_mesh((sx * 0.0028, y, _z + 0.00005), (0.0003, 0.0059, 0.0001), SILK))
        out.append(_box_mesh((0.0, y + sx * 0.0028, _z + 0.00005), (0.0059, 0.0003, 0.0001), SILK))
    out.append(_box_mesh((-0.0037, y - 0.0037, _z + 0.00005), (0.0025, 0.0006, 0.0001), SILK, yaw=-math.pi / 4))
    # the (unused) switch mounting hole behind the footprint
    out.append(_disc_mesh((0.005, y, _z + 0.00005), 0.002, HOLE))
    return out


BOARD = part(
    "board", pos=(*B_XY, TOP), fixed=True, friction=0.8,
    boxes=[
        ((0.0, 0.0, BASE_T / 2), (0.070, 0.130, BASE_T), ABS_BLACK),  # the holder's base plate
        *[((sx * (B_WID / 2 + RAIL_T / 2), 0.0, BASE_T + RAIL_H / 2), (RAIL_T, 0.110, RAIL_H), ALU) for sx in (-1, 1)],
        *[((sx * (B_WID / 2 - 0.0015), 0.0, LEDGE_Z - 0.001), (0.003, 0.110, 0.002), ALU) for sx in (-1, 1)],  # ledges
        ((0.0, 0.0, LEDGE_Z + B_T / 2), (B_WID, B_LEN, B_T), FR4),  # the board itself
    ],
    meshes=[m for y in FP_Y for m in _footprint(y)],
)
FP_C = [np.array([B_XY[0], B_XY[1] + y, B_TOP]) for y in FP_Y]  # footprint centres on the board's top face (world)
BOARD_C = np.array([*B_XY, B_TOP])

# -- the trays (fixed) and the LEDs (loose) ---------------------------------------------------------------------------
TRAY_T = 0.003
TRAY_TOP = TOP + TRAY_T
L_TRAY_XY, R_TRAY_XY = (0.19, 0.17), (0.19, -0.17)
LED_Z0 = TRAY_TOP + LED_H / 2 + 0.0002
# name: (tray x, tray y, yaw of the LED's cut corner from the correct orientation, degrees ccw from above)
LED_START = {
    "led_1": (0.19, 0.19, 60.0),
    "led_2": (0.19, 0.17, -45.0),
    "led_3": (0.19, 0.15, 30.0),
    "led_4": (0.19, -0.16, -60.0),
    "led_5": (0.19, -0.18, 45.0),
}
LED_NAMES = list(LED_START)
LED_ARM = {"led_1": "left", "led_2": "left", "led_3": "left", "led_4": "right", "led_5": "right"}
LED_FP = {"led_1": 0, "led_2": 1, "led_3": 2, "led_4": 3, "led_5": 4}  # the footprint the demo gives each LED


def _tray(name, xy, leds):
    meshes = []
    for n in leds:
        x, y, yaw = LED_START[n]
        lx, ly = x - xy[0], y - xy[1]
        for sx in (-1, 1):  # a white 6 mm square round the part's place
            meshes.append(_box_mesh((lx + sx * 0.003, ly, TRAY_T + 0.00005), (0.0003, 0.0063, 0.0001), SILK))
            meshes.append(_box_mesh((lx, ly + sx * 0.003, TRAY_T + 0.00005), (0.0063, 0.0003, 0.0001), SILK))
        a = CORNER_DIR + math.radians(yaw)  # a printed tick where the cut corner points
        meshes.append(_box_mesh((lx + 0.0045 * math.cos(a), ly + 0.0045 * math.sin(a), TRAY_T + 0.00005),
                                (0.002, 0.0005, 0.0001), SILK, yaw=a))
    return part(name, pos=(*xy, TOP), fixed=True, friction=0.9,
                boxes=[((0.0, 0.0, TRAY_T / 2), (0.030, 0.070, TRAY_T), ESD_BLACK)], meshes=meshes)


TRAY_L = _tray("tray_l", L_TRAY_XY, ["led_1", "led_2", "led_3"])
TRAY_R = _tray("tray_r", R_TRAY_XY, ["led_4", "led_5"])


def _led(name):
    x, y, yaw = LED_START[name]
    e = (LED_W - MARK_W) / 2
    return part(name, pos=(x, y, LED_Z0), yaw=math.radians(yaw), mass=LED_KG, friction=1.1,
                boxes=[((0.0, 0.0, 0.0), (LED_W, LED_W, LED_H), LED_BODY)],
                meshes=[_box_mesh((0.0, 0.0, LED_H / 2 + 0.00005), (0.0040, 0.0040, 0.0001), PHOSPHOR),
                        _box_mesh((-e, -e, LED_H / 2 + 0.00012), (MARK_W, MARK_W, 0.0001), MARK)])


LEDS = [_led(n) for n in LED_NAMES]
LED0 = {n: np.array([LED_START[n][0], LED_START[n][1], LED_Z0]) for n in LED_NAMES}

# -- the tweezers (loose), two pairs, standing tips down in cups: straight flat-tip tweezers. Frame: origin between
# the blades' tips, the arms vertical (joined end up), the flat blades the lower 12 mm of the arms. The -x arm and
# blade are the body; the +x arm and blade slide together along x (sprung open), so the jaw closes horizontally while
# the tweezers hang straight down from the gripper. ----------------------------------------------------------------------
BLADE_L, BLADE_W, BLADE_T = 0.012, 0.008, 0.0012
OPEN_GAP = 0.009  # the blades' gap at rest: room round a 5 mm part
ARM_L, ARM_W, ARM_T = 0.085, 0.008, 0.0025
SEP = OPEN_GAP + BLADE_T  # the two halves' centre planes
TOP_Z = BLADE_L + ARM_L


def _half(sx):
    return [((sx, 0.0, BLADE_L + ARM_L / 2), (ARM_T, ARM_W, ARM_L), STEEL),
            ((sx, 0.0, BLADE_L / 2), (BLADE_T, BLADE_W, BLADE_L), STEEL)]


CUP_FLOOR, CUP_H = 0.020, 0.035
CUP_X = (-(SEP / 2 + ARM_T / 2 + 0.0015), SEP / 2 + ARM_T / 2 + 0.0015)  # the pocket's x span
CUP_HY = ARM_W / 2 + 0.002
_cw = 0.003


def _cup(name, xy):
    cx, w = sum(CUP_X) / 2, CUP_X[1] - CUP_X[0]
    return part(name, pos=(*xy, TOP), fixed=True, friction=0.8,
                boxes=[((cx, 0.0, CUP_FLOOR / 2), (w + 2 * _cw, 2 * CUP_HY + 2 * _cw, CUP_FLOOR), CUP_C),
                       *[((x, 0.0, CUP_FLOOR + CUP_H / 2), (_cw, 2 * CUP_HY + 2 * _cw, CUP_H), CUP_C)
                         for x in (CUP_X[0] - _cw / 2, CUP_X[1] + _cw / 2)],
                       *[((cx, e * (CUP_HY + _cw / 2), CUP_FLOOR + CUP_H / 2), (w, _cw, CUP_H), CUP_C)
                         for e in (-1, 1)]])


def _tweezers(name, xy):
    jaw = Joint("jaw", kind="slide", axis="X", anchor=(SEP / 2, 0.0, BLADE_L + ARM_L / 2),
                lower=-(SEP - BLADE_T - 0.0002), upper=0.0, initial=0.0, target=0.0, stiffness=600.0,
                damping=5.0, mass=0.008, boxes=_half(SEP / 2))
    return part(name, pos=(*xy, TOP + CUP_FLOOR + 0.0012), mass=0.020, friction=1.1, boxes=_half(-SEP / 2),
                meshes=[_box_mesh((0.0, 0.0, TOP_Z - 0.004), (SEP - ARM_T, ARM_W, 0.008), STEEL)], joints=[jaw])


L_CUP_XY, R_CUP_XY = (0.12, 0.25), (0.11, -0.215)
CUP_L, CUP_R = _cup("cup_l", L_CUP_XY), _cup("cup_r", R_CUP_XY)
TW_L, TW_R = _tweezers("tweezers_l", L_CUP_XY), _tweezers("tweezers_r", R_CUP_XY)
TW_HOME = {"left": np.array([*L_CUP_XY, TOP + CUP_FLOOR + 0.0012]),
           "right": np.array([*R_CUP_XY, TOP + CUP_FLOOR + 0.0012])}
TW_NAME = {"left": "tweezers_l", "right": "tweezers_r"}

OBJECTS = [BOARD, TRAY_L, TRAY_R, *LEDS, CUP_L, CUP_R, TW_L, TW_R]

# -- scoring ----------------------------------------------------------------------------------------------------------
POS_TOL = 0.001  # the LED's centre within 1 mm of the footprint's centre
YAW_TOL = 10.0  # its cut corner within 10 degrees of the corner bar's direction
TILT_TOL = 5.0  # lying flat: tilted at most 5 degrees
SEAT_TOL = 0.0004  # its centre within 0.4 mm of a part lying on the pads
HANDS_FAR = 0.08  # both hands' pad tips at least this far from the board's centre at the end
TOOLS_FAR = 0.03  # ... and at least this far from either pair of tweezers' centre line: the hands have let go of them
TOUCH = 0.0015  # a pad tip within 1.5 mm of an LED's body: the LED touched by the gripper itself
CUP_XY_TOL, CUP_Z_TOL, CUP_TILT = 0.010, 0.004, 15.0  # tweezers standing in their cup
SETTLE_STEPS, SETTLE_TOL = 20, 0.001  # every LED moved under 1 mm over the last second (20 steps at 20 Hz)

_state = {}
_hist = []


@on_reset
def _reset_state(env):
    _state.update(touched=False, far=True, off_tools=True, settled=False)
    _hist.clear()


def _tips(env):
    """Both grippers' pad tips (world), two per hand."""
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7"], preserve_order=True)[0]
        half = abs(float(robot.data.joint_pos.torch[0, ids].cpu().numpy()[0]))
        out += [mid + half * g[:3, 0], mid - half * g[:3, 0]]
    return out


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def corner_dir(led) -> float:
    """Where the LED's cut corner points, seen from above (rad)."""
    v = led.rotation() @ np.array([-1.0, -1.0, 0.0])
    return math.atan2(v[1], v[0])


def yaw_error_deg(led, fp_yaw=FP_YAW) -> float:
    return abs(math.degrees(_wrap(corner_dir(led) - (CORNER_DIR + fp_yaw))))


def placed_on(led, fp_c) -> bool:
    """Flat on the footprint, centred within 1 mm, cut corner within 10 degrees of the corner bar."""
    p = np.asarray(led.pos, float)
    return (np.hypot(*(p[:2] - fp_c[:2])) <= POS_TOL and abs(p[2] - (fp_c[2] + LED_H / 2)) <= SEAT_TOL
            and led.tilt_deg() <= TILT_TOL and yaw_error_deg(led) <= YAW_TOL)


def placed_count(objects) -> int:
    """Footprints holding a correctly placed LED; each LED and each footprint counts once."""
    taken = set()
    n = 0
    for name in LED_NAMES:
        led = objects[name]
        p = np.asarray(led.pos, float)
        i = int(np.argmin([np.hypot(*(p[:2] - c[:2])) for c in FP_C]))
        if i not in taken and placed_on(led, FP_C[i]):
            taken.add(i)
            n += 1
    return n


def tweezers_home(objects) -> bool:
    for arm, name in TW_NAME.items():
        t = objects[name]
        p = np.asarray(t.pos, float)
        if (np.hypot(*(p[:2] - TW_HOME[arm][:2])) > CUP_XY_TOL or abs(p[2] - TW_HOME[arm][2]) > CUP_Z_TOL
                or t.tilt_deg() > CUP_TILT):
            return False
    return True


def tweezer_clearance(tips, objects) -> float:
    """The smallest distance (m) from any pad tip to either pair of tweezers' centre line, the segment from between
    the blades' tips up to the top of the arms."""
    out = math.inf
    for name in TW_NAME.values():
        t = objects[name]
        a = np.asarray(t.pos, float)
        ab = t.rotation() @ np.array([0.0, 0.0, TOP_Z])
        for tip in tips:
            d = np.asarray(tip, float) - a
            s = float(np.clip(np.dot(d, ab) / np.dot(ab, ab), 0.0, 1.0))
            out = min(out, float(np.linalg.norm(d - s * ab)))
    return out


@on_step
def _watch(env):
    """Latch a gripper pad touching an LED; track whether the hands are clear of the board and of the tweezers, and
    the LEDs still."""
    S = labkit._SPECS
    tips = _tips(env)
    tw = {n: labkit._part_state(env, S[n]) for n in TW_NAME.values()}
    _state["off_tools"] = tweezer_clearance(tips, tw) >= TOOLS_FAR
    track = []
    for n in LED_NAMES:
        led = labkit._part_state(env, S[n])
        p = np.asarray(led.pos, float)
        R = led.rotation()
        for tip in tips:
            q = np.abs(R.T @ (np.asarray(tip, float) - p))
            if q[0] < LED_W / 2 + TOUCH and q[1] < LED_W / 2 + TOUCH and q[2] < LED_H / 2 + TOUCH:
                _state["touched"] = True
        track.append(p)
    _state["far"] = all(np.hypot(*(np.asarray(t)[:2] - BOARD_C[:2])) >= HANDS_FAR for t in tips)
    _hist.append(np.concatenate(track))
    del _hist[:-(SETTLE_STEPS + 1)]
    _state["settled"] = (len(_hist) > SETTLE_STEPS
                         and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 if a gripper pad ever touched an LED, or the LEDs have not all been still (under 1 mm) for the last second.
    Otherwise 0.2 per footprint holding a correctly placed LED (flat, centred within 1 mm, cut corner within
    10 degrees of the corner bar); 1.0 only when all five are placed, both tweezers stand in their cups, both hands
    are 8 cm from the board's centre and have let go of the tweezers (every pad tip 3 cm from them), else at most
    0.8."""
    if _state.get("touched") or not _state.get("settled", False):
        return 0.0
    n = placed_count(objects)
    if n == 5 and tweezers_home(objects) and _state.get("far", True) and _state.get("off_tools", True):
        return 1.0
    return min(0.2 * n, 0.8)


SUCCESS_HOLD_S = 1.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """Mean distance (m) from each footprint's centre to the nearest LED."""
    ps = [np.asarray(objects[n].pos, float) for n in LED_NAMES]
    return float(np.mean([min(np.linalg.norm(p - c) for p in ps) for c in FP_C]))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Each hand takes its tweezers out of their cup and holds them hanging straight down. For each of its LEDs it
# straddles the part with the open blades, pinches it by two opposite sides, lifts it, turns it by the angle its cut
# corner is off, carries it over its footprint, lines it up (watching the part), sets it on the pads and lets go. Then it stands the tweezers back in their cup. The left hand places three LEDs, the right two; the hands
# take turns over the board, one placement at a time. The demo reads only what the cameras see: the tweezers' tips
# (to trim each hand's aim) and the LEDs.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._rot = True
    _demo.grip_rotation = _grip

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
TIP_ABOVE_TCP = 0.013
PAD_IN = 0.006  # the pads hold the tweezers' arms 6 mm in from the pad tips
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.15, -0.25, TOP + 0.16)
GRIP_X = _prev_grip(math.pi / 2)  # a straight-down grip whose pads close along the tweezers' x (the jaw's axis)
TW_GRIP_F = np.array([0.0, 0.0, TOP_Z - 0.014])  # where the pads hold the arms (tweezers frame): 14 mm below their top
# finger targets: round the arms; holding them; squeezed. The fingers are stiff position drives (5000 N/m), so the
# pinch target sits only about 1 mm past the blades meeting the part (about 1.5 N on it); a deeper target drives the
# blade through a 0.3 g part instead of squeezing it.
TW_WIDE, TW_HOLD, TW_PINCH = -0.014, -0.0056, -0.0034
PINCH_DX = -(SEP - BLADE_T - LED_W) / 2  # a pinched LED's centre from the tips' origin, along the tweezers' x
# The blades close with their tips at the tray (the LED's centre 0.8 mm above them). The arm's joints have friction
# and soft drives, so a hand stops up to about 1 mm from its aim: aimed at the surface, a hand that stops high still
# covers 0.6 mm of the part's side, and one that stops low only presses lightly on the tray.
LED_ABOVE_TIPS = 0.0008
SET_PRESS = 0.0002  # set the part down this far into the pads (the soft arm presses lightly) rather than dropping it
V_LED = np.array([PINCH_DX, 0.0, LED_ABOVE_TIPS])  # pinched: the part against the fixed blade
# picking: the open blades centred on the part. Holding the arms (TW_HOLD) already closes the jaw about 1.1 mm, so the
# held gap (7.9 mm) is centred 0.56 mm towards the fixed blade: about 1.4 mm clear on each side of the part.
HELD_GAP_DX = -0.00056
V_PICK = np.array([HELD_GAP_DX, 0.0, LED_ABOVE_TIPS])
# Closed-loop trims, per control step while the thing watched is still: a fraction of the seen error, at most 0.3 mm.
# The arm takes about 0.4 s to follow a new aim, so a larger fraction keeps adding what it has not yet done and the
# aim swings by +-2 mm; 0.08 settles in about a second.
TRIM_GAIN = 0.08
# Preferred jaw heading: -90 degrees puts the jaw along world y, which is the gripper's default (yaw 0) grip; the
# picks then stay within 60 degrees of it and every placement happens exactly at it.
H_PREF = {"left": math.radians(float(os.environ.get("TW_HEAD_L", "-90"))),
          "right": math.radians(float(os.environ.get("TW_HEAD_R", "-90")))}
VARIANT = os.environ.get("DEMO_VARIANT", "")  # "" (full), "hold" (tweezers kept out), "norotate" (no turning)


def Rz(a):
    return axis_angle_matrix(Z, a)


def _tcp(point, rot):
    a = rot[:, 2]
    return np.asarray(point, float) - a * (TIP_ABOVE_TCP - PAD_IN)


def _tw_pose_tcp(origin, h):
    """TCP for a hand holding the tweezers hanging straight down, their jaw along heading ``h``, tips at ``origin``."""
    rot = Rz(h) @ GRIP_X
    return _tcp(np.asarray(origin, float) + Rz(h) @ TW_GRIP_F, rot), Orient(rot)


_cup_tcp = _tw_pose_tcp  # in the cup the tweezers hang the same way, jaw along +x


def _headings(theta_deg, delta_deg, h_pref):
    """The pick heading (a side pair of the LED, mod 90 degrees) and the place heading, keeping both near ``h_pref``."""
    best = None
    for k in range(4):
        h = math.radians(theta_deg + 90.0 * k)
        hp = h + math.radians(delta_deg)
        cost = max(abs(_wrap(h - h_pref)), abs(_wrap(hp - h_pref)))
        if best is None or cost < best[0]:
            best = (cost, _wrap(h), _wrap(hp))
    return best[1], best[2]


# what each hand's tweezers are seen doing: the tips' live position; each hand's aim is trimmed by the offset between
# where it means them to be and where they are, integrated only while they are still and near it
_tw = {s: {"corr": np.zeros(3), "goal": None, "prev": None} for s in ("left", "right")}
_seen = {}


def _clear_demo_state():
    """Forget what was seen; keep the plan's schedule (the ``_pick_t`` and ``_win`` times ``demo()`` sets)."""
    for d in _tw.values():
        d.update(corr=np.zeros(3), goal=None, prev=None)
    for n in LED_NAMES:
        _seen.pop(n + "_pick", None)
        _seen.pop(n + "_prev", None)
        _seen.pop(n + "_off", None)
        _seen[n + "_trim"] = np.zeros(2)


@on_reset
def _reset_demo(env):
    _clear_demo_state()


@on_step
def _track(env):
    S = labkit._SPECS
    for s, n in TW_NAME.items():
        d = _tw[s]
        pos = np.asarray(labkit._part_state(env, S[n]).pos, float)
        prev, d["prev"] = d["prev"], pos
        if d["goal"] is None or prev is None:
            continue
        err = d["goal"] - pos
        step = pos - prev
        if d.get("mode") == "xy":  # a straight move down: its aim across doesn't change, so trim only that
            err[2] = 0.0
            still = np.linalg.norm(step[:2]) < 0.0002  # not drifting sideways
        else:  # a hand holding still at a waypoint
            still = np.linalg.norm(step) < 0.0002
        if still and np.linalg.norm(err) < 0.04:
            d["corr"] = np.clip(d["corr"] + np.clip(TRIM_GAIN * err, -0.0003, 0.0003), -0.03, 0.03)
    t = getattr(_demo, "_plan_clock", 0.0)
    for n in LED_NAMES:
        led = labkit._part_state(env, S[n])
        p = np.asarray(led.pos, float)
        prev, _seen[n + "_prev"] = _seen.get(n + "_prev"), p
        if t < _seen.get(n + "_pick_t", 1e9):  # the part as seen before the blades close round it
            _seen[n + "_pick"] = p
        held = _seen.get(n + "_held")  # carried in the blades: where the part sits relative to the tweezers' tips
        if held and held[0] <= t < held[1]:
            _seen[n + "_off"] = labkit._part_state(env, S[TW_NAME[LED_ARM[n]]]).to_local(p)
        win = _seen.get(n + "_win")  # lining it up over its footprint: trim the aim by where the part is seen
        if win and win[0] <= t < win[1] and prev is not None and np.linalg.norm(p - prev) < 0.0002:
            aim = FP_C[LED_FP[n]][:2]
            tr = _seen[n + "_trim"]
            tr[:] = np.clip(tr + np.clip(TRIM_GAIN * (aim - p[:2]), -0.0003, 0.0003), -0.005, 0.005)


class _At:
    """A hand's TCP from a function giving (tweezer tips, heading) or, with ``cup``, (tips, yaw) upright; trimmed by
    that hand's live correction (held fixed when ``track`` is False)."""

    def __init__(self, side, fn, track=True, cup=False):
        self.side, self.fn, self.track, self.cup = side, fn, track, cup

    def pose(self):
        o, h = self.fn()
        tcp, rot = _cup_tcp(o, h) if self.cup else _tw_pose_tcp(o, h)
        return o, tcp, rot

    def __array__(self, dtype=None, copy=None):
        o, tcp, _ = self.pose()
        d = _tw[self.side]
        d["goal"] = np.asarray(o, float) if self.track else None
        d["mode"] = self.track
        return np.asarray(tcp + d["corr"], dtype=dtype or float)


def _wp(dur, side, fn, finger, label, track=True, cup=False):
    at = _At(side, fn, track, cup)
    return Waypoint(dur, at, finger, label, at.pose()[2])


def _const(o, h):
    o = np.asarray(o, float)
    return lambda: (o, h)


def _dur(wps):
    return sum(w.duration for w in wps)


def _hold_until(wps, t):
    """Hold the last pose until plan time ``t``."""
    dt = t - _dur(wps)
    if dt > 0.05:
        last = wps[-1]
        wps.append(Waypoint(dt, last.pos, last.finger, "wait", last.yaw))


def _cup_out(side):
    home = TW_HOME[side]
    return [
        _wp(3.0, side, _const(home + Z * 0.05, 0.0), TW_WIDE, "above the tweezers in their cup", False, True),
        _wp(1.5, side, _const(home, 0.0), TW_WIDE, "down: pads either side of their arms", False, True),
        _wp(0.8, side, _const(home, 0.0), TW_HOLD, "hold them, blades open", False, True),
        _wp(1.5, side, _const(home + Z * 0.06, 0.0), TW_HOLD, "lift them out", False, True),
    ]


def _cup_in(side):
    # The arms clear the cup's pocket by about 1.5 mm a side, and the hand drifts sideways on a long move down: line
    # them up again just above the rim, then lower them in, trimming across all the way down.
    home = TW_HOME[side]
    rim = CUP_H - 0.0012 + 0.002  # the tips 2 mm above the cup's rim (home is 1.2 mm above its floor)
    return [
        _wp(3.0, side, _const(home + Z * 0.06, 0.0), TW_HOLD, "upright again, over the cup", False, True),
        _wp(1.5, side, _const(home + Z * 0.06, 0.0), TW_HOLD, "line them up over the cup", True, True),
        _wp(1.0, side, _const(home + Z * rim, 0.0), TW_HOLD, "down to the cup's rim", "xy", True),
        _wp(1.0, side, _const(home + Z * rim, 0.0), TW_HOLD, "line them up at the rim", True, True),
        _wp(1.2, side, _const(home + Z * 0.002, 0.0), TW_HOLD, "lower them into the cup", "xy", True),
        _wp(0.6, side, _const(home + Z * 0.002, 0.0), TW_WIDE, "let go", False, True),
        _wp(1.2, side, _const(home + Z * 0.05, 0.0), TW_WIDE, "up", False, True),
    ]


def _pick_place(side, name, wps, place_at=None):
    """Append one LED's pick and place to ``wps``; ``place_at`` holds the part in the air until that plan time."""
    x0, y0, theta = LED_START[name]
    delta = math.degrees(_wrap(math.radians(FP_YAW - theta)))  # how far the part must turn
    h, hp = _headings(theta, delta, H_PREF[side])
    if VARIANT == "norotate":
        hp = h
    fp = FP_C[LED_FP[name]]

    def pick(up):
        def f():
            c = _seen.get(name + "_pick", LED0[name])
            return c - Rz(h) @ V_PICK + Z * up, h
        return f

    def place(up):
        def f():
            tr = _seen.get(name + "_trim", np.zeros(2))
            off = _seen.get(name + "_off", V_LED)  # the part in the blades as seen while carrying it (else as planned)
            c = fp + np.array([tr[0], tr[1], LED_H / 2 - SET_PRESS])  # the part's centre, set on the pads
            return c - Rz(hp) @ off + Z * up, hp
        return f

    # The tips are trimmed only while the hand holds still at a waypoint ("line up"): a moving waypoint's target is
    # its end pose, so trimming while moving would learn the path still to go as an error.
    wps += [
        _wp(3.0, side, pick(0.03), TW_HOLD, f"{name}: above it, blades pointing down", False),
        _wp(1.2, side, pick(0.004), TW_HOLD, f"{name}: down, blades either side of it", False),
        _wp(1.6, side, pick(0.004), TW_HOLD, f"{name}: line up"),
    ]
    _seen[name + "_pick_t"] = _dur(wps)  # the part, as seen before the blades go round it
    wps += [
        # no trimming on the last 4 mm: a correction that lands while the blades straddle the part can catch it
        _wp(0.8, side, pick(0.0), TW_HOLD, f"{name}: tips down at the tray", False),
        _wp(0.8, side, pick(0.0), TW_PINCH, f"{name}: pinch it", False),
        _wp(1.2, side, pick(0.03), TW_PINCH, f"{name}: lift it", False),
    ]
    t_lifted = _dur(wps)
    if place_at is not None:
        _hold_until(wps, place_at)
    _seen[name + "_place_t"] = _dur(wps)  # when this hand starts towards the board with it
    _seen[name + "_held"] = (t_lifted - 0.2, _seen[name + "_place_t"] + 0.3)  # watch the part in the blades
    wps += [_wp(2.5, side, place(0.03), TW_PINCH, f"{name}: carry it over its footprint, turning it", False)]
    # Line up 3 mm above the pads, so the last move down is short (the arm drifts sideways by up to about 1 mm on a
    # long one), set it down, and keep trimming while it rests on the pads in the blades: the pinched part is slid
    # onto its footprint before they open.
    t0 = _dur(wps)
    wps += [
        _wp(1.5, side, place(0.003), TW_PINCH, f"{name}: down over the footprint", False),
        _wp(3.0, side, place(0.003), TW_PINCH, f"{name}: line it up, seen against the silkscreen", False),
        _wp(0.8, side, place(0.0), TW_PINCH, f"{name}: set it down on the pads", False),
        _wp(2.0, side, place(0.0), TW_PINCH, f"{name}: nudge it onto the footprint", False),
    ]
    _seen[name + "_win"] = (t0 + 0.8, _dur(wps))
    wps += [
        _wp(0.6, side, place(0.0), TW_HOLD, f"{name}: let go: open the blades", False),
        _wp(0.8, side, place(0.02), TW_HOLD, f"{name}: up", False),
    ]
    return wps


# The hands take turns over the board, one placement per slot: at placement both grippers close along y and each
# housing spans about +-51 mm along that axis, more than the footprints' spacing, so two hands over the board at once
# would meet. A placement (carry to up) takes 11.2 s; the rest of the slot lets the hand clear the board.
T_SLOT = 12.5
PLACE_ORDER = [("left", "led_1"), ("right", "led_4"), ("left", "led_2"), ("right", "led_5"), ("left", "led_3")]


def demo():
    _clear_demo_state()
    plans = {"left": _cup_out("left"), "right": _cup_out("right")}
    t0 = None
    for i, (side, name) in enumerate(PLACE_ORDER):
        # each hand picks its next LED as soon as it is free, then holds it in the air until its slot
        _pick_place(side, name, plans[side], place_at=None if t0 is None else t0 + i * T_SLOT)
        if t0 is None:
            t0 = _seen[name + "_place_t"]
    left, right = plans["left"], plans["right"]
    if VARIANT != "hold":
        left += _cup_in("left")
        right += _cup_in("right")
        left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"))
        right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    t_end = max(_dur(left), _dur(right)) + 2.0
    for wps in (left, right):
        last = wps[-1]
        wps.append(Waypoint(t_end - _dur(wps), last.pos, last.finger, "rest", last.yaw))
    return {"left": left, "right": right}
