"""Checks of task.py's scene, scoring and demo without the simulator. Needs yam_arena (the official stack) and the
labkit.py next to task.py:

    python -m pytest -q test_task.py

The reach test runs only where yam_arena's IK is importable (the Arena venv); the rest runs anywhere.
"""

import importlib.util
import math
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task as T  # noqa: E402
from labkit import PartState  # noqa: E402


def _quat(rot):
    """xyzw from a rotation matrix."""
    m = rot
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def state(pos, yaw_deg=0.0, tilt_deg=0.0, size=(0.005, 0.005, 0.0016), vel=0.0):
    rot = _rz(math.radians(yaw_deg)) @ _rx(math.radians(tilt_deg))
    pos = np.asarray(pos, float)
    half = np.asarray(size, float) / 2
    return PartState(pos=pos, quat_xyzw=_quat(rot), lin_vel=np.array([vel, 0.0, 0.0]), ang_vel=np.zeros(3),
                     low=pos - half, high=pos + half)


def led_on(i, dx=0.0, dy=0.0, dz=0.0, yaw_deg=0.0, tilt_deg=0.0):
    """An LED lying on footprint ``i`` (correct when the offsets are 0)."""
    c = T.FP_C[i]
    return state((c[0] + dx, c[1] + dy, c[2] + T.LED_H / 2 + dz), yaw_deg, tilt_deg)


def led_start(name):
    x, y, yaw = T.LED_START[name]
    return state((x, y, T.LED_Z0), yaw)


def tweezers(arm, home=True):
    p = T.TW_HOME[arm].copy()
    if not home:
        p += [0.10, 0.0, 0.08]
    return state(p, 0.0, 0.0 if home else 45.0, size=(0.03, 0.01, 0.1))


def scene(**over):
    """Every LED placed, tweezers in their cups, unless overridden by name."""
    objs = {n: led_on(i) for i, n in enumerate(T.LED_NAMES)}
    objs["tweezers_l"], objs["tweezers_r"] = tweezers("left"), tweezers("right")
    for fixed in ("board", "tray_l", "tray_r", "cup_l", "cup_r"):
        objs[fixed] = state((0.0, 0.0, 0.0), size=(0.1, 0.1, 0.01))
    objs.update(over)
    return objs


@pytest.fixture(autouse=True)
def clean_state():
    T._state.update(touched=False, far=True, off_tools=True, settled=True)
    yield
    T._state.update(touched=False, far=True, off_tools=True, settled=True)


# -- scene -------------------------------------------------------------------------------------------------------------

def test_the_scene_has_the_right_objects():
    names = [o.name for o in T.OBJECTS]
    assert len(names) == len(set(names)) == 12  # board, 2 trays, 5 LEDs, 2 cups, 2 tweezers
    assert all(n.isidentifier() for n in names)
    loose = {o.name for o in T.OBJECTS if o.mass is not None}
    assert loose == {*T.LED_NAMES, "tweezers_l", "tweezers_r"}
    for o in T.OBJECTS:
        if o.name.startswith("tweezers"):
            assert len(o.joints) == 1 and o.joints[0].kind == "slide"


def test_the_leds_start_on_their_trays_at_the_stated_angles():
    assert [T.LED_START[n][2] for n in T.LED_NAMES] == [60.0, -45.0, 30.0, -60.0, 45.0]
    for n in T.LED_NAMES:
        x, y, _ = T.LED_START[n]
        tray = T.L_TRAY_XY if T.LED_ARM[n] == "left" else T.R_TRAY_XY
        assert abs(x - tray[0]) <= 0.015 and abs(y - tray[1]) <= 0.035 - 0.003


def test_footprints_follow_the_key_pitch():
    ys = [c[1] for c in T.FP_C]
    assert np.allclose(np.diff(ys), -0.01905)
    assert all(abs(c[0] - 0.30) < 1e-9 and abs(c[2] - T.B_TOP) < 1e-9 for c in T.FP_C)


# -- scoring -----------------------------------------------------------------------------------------------------------

def test_all_five_placed_and_released_scores_one():
    objs = scene()
    assert T.placed_count(objs) == 5
    assert T.score(objs) == 1.0 and T.success(objs)


def test_partial_credit_counts_placed_leds():
    objs = scene(led_4=led_start("led_4"), led_5=led_start("led_5"))
    assert T.score(objs) == pytest.approx(0.6)
    objs = scene(**{n: led_start(n) for n in T.LED_NAMES})
    assert T.score(objs) == 0.0


def test_full_credit_needs_the_tweezers_in_their_cups_let_go_and_hands_clear():
    assert T.score(scene(tweezers_l=tweezers("left", home=False))) == pytest.approx(0.8)
    T._state["far"] = False
    assert T.score(scene()) == pytest.approx(0.8)
    T._state.update(far=True, off_tools=False)  # a hand still holding its tweezers, standing in their cup
    assert T.score(scene()) == pytest.approx(0.8)
    # the clearance: pad tips where a hand holds the arms are on the tweezers; 3.5 cm above their top they are clear
    objs = scene()
    grip = T.TW_HOME["left"] + T.TW_GRIP_F
    top = T.TW_HOME["left"] + np.array([0.0, 0.0, T.TOP_Z])
    assert T.tweezer_clearance([grip + [0.007, 0.0, 0.0], grip - [0.007, 0.0, 0.0]], objs) < T.TOOLS_FAR
    assert T.tweezer_clearance([T.TW_HOME["right"] + [0.0, 0.0, 0.05]], objs) < T.TOOLS_FAR  # both pairs count
    assert T.tweezer_clearance([top + [0.0, 0.0, 0.035]], objs) >= T.TOOLS_FAR
    assert T.tweezer_clearance([top + [0.0, 0.0, 0.025]], objs) < T.TOOLS_FAR


@pytest.mark.parametrize("kw", [
    dict(yaw_deg=15.0),  # cut corner 15 degrees off
    dict(yaw_deg=90.0),  # right square, wrong corner
    dict(yaw_deg=180.0),  # reversed
    dict(dx=0.0015),  # 1.5 mm off centre
    dict(dy=-0.0012),
    dict(tilt_deg=180.0),  # upside down
    dict(tilt_deg=8.0),  # propped on a pad edge
    dict(dz=0.0016),  # sitting on another LED
])
def test_a_misplaced_led_earns_nothing(kw):
    objs = scene(led_3=led_on(2, **kw))
    assert T.placed_count(objs) == 4
    assert T.score(objs) == pytest.approx(0.8)


def test_tolerances_are_inclusive_where_stated():
    objs = scene(led_3=led_on(2, dx=0.0009, yaw_deg=9.0, tilt_deg=4.0))
    assert T.placed_count(objs) == 5


def test_two_leds_on_one_footprint_count_once():
    objs = scene(led_3=led_on(1, dx=0.0006))  # beside led_2 on footprint 1
    assert T.placed_count(objs) == 4


def test_latched_violations_and_motion_score_zero():
    T._state["touched"] = True
    assert T.score(scene()) == 0.0
    T._state["touched"] = False
    T._state["settled"] = False
    assert T.score(scene()) == 0.0


def test_distance_is_zero_when_placed_and_positive_at_the_start():
    assert T.distance(scene()) == pytest.approx(T.LED_H / 2)
    assert T.distance(scene(**{n: led_start(n) for n in T.LED_NAMES})) > 0.1


def test_corner_direction_convention():
    led = led_on(0)  # correct orientation: corner towards -x, -y
    assert T.yaw_error_deg(led) == pytest.approx(0.0, abs=1e-6)
    assert T.yaw_error_deg(led_on(0, yaw_deg=-45.0)) == pytest.approx(45.0)


# -- demo --------------------------------------------------------------------------------------------------------------

def test_headings_turn_each_led_by_its_offset_and_stay_near_the_preferred_heading():
    for n, (x, y, theta) in T.LED_START.items():
        side = T.LED_ARM[n]
        delta = math.degrees(T._wrap(math.radians(T.FP_YAW - theta)))
        h, hp = T._headings(theta, delta, T.H_PREF[side])
        assert abs(math.degrees(T._wrap(hp - h)) - delta) < 1e-9
        assert abs(delta) <= 60.0
        assert abs(math.degrees(T._wrap(h - T.H_PREF[side]))) <= 90.0 + 1e-9
        assert abs(math.degrees(T._wrap(hp - T.H_PREF[side]))) <= 90.0 + 1e-9


def test_demo_plan_shape_and_timing():
    plan = T.demo()
    assert set(plan) == {"left", "right"}
    left, right = plan["left"], plan["right"]
    assert abs(T._dur(left) - T._dur(right)) < 1e-6
    assert T._dur(left) <= 110.0, "must finish well inside 2,400 steps at 20 Hz"
    labels = [w.label for w in left]
    assert any("led_1" in s for s in labels) and any("led_3" in s for s in labels)
    assert not any("led_4" in s for s in labels)
    # the schedule: the hands take turns over the board (carry to up, plus 1 s to clear it), each in its own slot
    busy = []
    for arm, wps in plan.items():
        t, start = 0.0, None
        for w in wps:
            if ": carry" in w.label:
                start = t
            t += w.duration
            if start is not None and w.label.endswith(": up"):
                busy.append((start, t + 1.0, arm, w.label.split(":")[0]))
                start = None
    busy.sort()
    assert [b[3] for b in busy] == [n for _, n in T.PLACE_ORDER]
    for (s0, e0, a0, _), (s1, e1, a1, _) in zip(busy, busy[1:]):
        assert s1 >= e0, f"{a0} and {a1} would be over the board together ({s1:.1f} < {e0:.1f})"
    starts = [b[0] for b in busy]
    assert np.allclose(np.diff(starts), T.T_SLOT), "every placement starts on its slot (no hand arrives late)"


def test_pinch_geometry_is_consistent_with_the_tweezers():
    assert T.SEP - T.BLADE_T - T.LED_W > 0.0, "the blades must close further than the part's width"
    assert T.OPEN_GAP >= T.LED_W + 0.002, "the open blades must clear the part with room for contact offsets"
    assert T.PINCH_DX == pytest.approx(-(T.SEP - T.BLADE_T - T.LED_W) / 2)


@pytest.mark.skipif(importlib.util.find_spec("yam_arena.ik") is None, reason="needs yam_arena's IK (the Arena venv)")
def test_the_demo_is_reachable_within_the_joint_limits():
    from yam_arena import ik, station
    from yam_arena.constants import CONTROL_HZ
    from yam_arena.demo import ArmPlan

    plan = T.demo()
    worst = {}
    for arm, wps in plan.items():
        ap = ArmPlan(list(wps), station.BASES[arm])
        actions = np.array([ap.action7((k + 1) / CONTROL_HZ)[0] for k in range(round(ap.duration * CONTROL_HZ) + 5)])
        worst[arm] = (ap.worst_ik_mm, ap.worst_ik_deg)
        assert np.all(actions[:, :6] >= ik.LOWER - 1e-9) and np.all(actions[:, :6] <= ik.UPPER + 1e-9), arm
    for arm, (mm, deg) in worst.items():
        assert mm < 0.5 and deg < 0.1, f"{arm}: IK error {mm:.2f} mm, {deg:.2f} deg"
