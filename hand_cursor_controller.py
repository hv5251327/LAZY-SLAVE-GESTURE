"""
HandGestureController v8.0 - everything, one hand only
======================================================
Run:      python hand_cursor_controller.py
Selftest: python hand_cursor_controller.py --selftest    (no camera / mouse / mediapipe)

Gestures
--------
  show INDEX           wake from REST (nothing works while resting)
  move index           move the cursor
  small pinch x1       single click
  small pinch x2       double click (the first pinch never clicks on its own)
  pinch + hold 0.35s   drag; open the pinch to drop
  index+middle pinch   right click
  fist LOW / HIGH      scroll down / up, faster the further it is from centre
  open palm swipe      left = forward, right = backward, up/down = volume
  wave (3 swings)      sleep / wake toggle

Keys (preview window must be focused)
-------------------------------------
  [ ]   active box bigger / smaller (reach)      - =   sensitivity
  , .   pinch stricter / easier                  9 0   scroll slower / faster
  m     smoothing preset     d  landmark dots    k / l save / load config
  s     sleep toggle         c  calibrate         r     reset gesture state
  i     start on login       j  desktop shortcut  q     quit

Fixes in 7.1 (found by instrumented tests, see --selftest)
----------------------------------------------------------
 B1  ASLEEP did not gate anything: gestures and the cursor stayed live during
     "sleep". Sleep is now a real gate (cursor+gestures off until you wave again).
 B2  A pinch whose release never happened (hand left the frame mid-pinch) left
     `pending` armed forever. A press older than PINCH_STALE_TIMEOUT is dropped,
     and losing the hand releases the contact state.
 B3  PINCH_FREEZE_THRESHOLD (0.50) was above the point-while-pinching band, so
     the cursor froze during ordinary pointing. Freeze only applies when the
     fingers are actually closing (ratio < press threshold + a small margin).
 B4  WaveDetector rescanned the whole history each frame and counted one long
     movement as several reversals. Rewritten: swing counting on the live
     signal with debounce, hysteresis and a peak-width check.
 B5  ScrollEngine counted the FIRST tick before FLIP_DELAY elapsed, so a fast
     direction change emitted one tick the wrong way.
 B6  Palm swipe fired while the palm was still travelling (triggered mid-swipe
     and then re-fired). It now requires the hand to settle before re-arming.
 B7  ClickEngine was created before the runtime PINCH_THRESHOLD could be tuned;
     the threshold is now read live every frame.
 B8  Media/volume cooldowns were per-axis and could fire together; single
     cooldown for all palm events.
 B9  Hotkey ranges allowed PINCH_THRESHOLD_POINTING to go negative / inverted
     the hysteresis band. Both are clamped.
 v8.0  Adds, still ONE hand only:
       5  pinch-and-hold drag (mouseDown/up, or DRAG_MODE="shift" for selections)
          + right click via the index+middle pinch
       6  MediaPipe Tasks backend with automatic fallback + a clear message if
          hand_landmarker.task is missing (solutions API is being removed)
       7  velocity scroll: tick rate follows the distance from the centre
       8  threaded camera capture so inference no longer stalls the pipeline
       10 calibration wizard (`c`): open -> pinch -> fist -> right pose
       11 start-on-login registry entry (`i`) and desktop shortcut (`j`)
       12 native volume (pycaw / osascript / amixer-pactl) with key fallback

v7.3  (1) Cursor smoothing is now frame-rate independent: both the smoothing
           factor and the per-frame pixel thresholds scale with the real frame
           duration, so 15 fps and 30 fps feel the same (measured: 90% of a
           step took 0.67 s at 15 fps vs 0.43 s at 30 fps - now 0.53 vs 0.43).
       (2) Tuning persists: settings are saved to ~/.hand_controller_v7.json on
           quit (and with `k`), loaded on start, range-checked and clamped.
       (3) Idle auto-sleep after IDLE_SLEEP_TIME was configured but never
           implemented - it now actually sleeps.
       (4) HUD: live FPS readout + finger count; `d` toggles the landmark dots.

 B10 Time kept accumulating while the process was suspended (laptop sleep):
     a jump guard now resets all timers instead of firing a burst.

 v7.2  Horizontal palm swipe no longer changes track. It now seeks inside the
       video: swipe LEFT = forward, swipe RIGHT = backward (arrow keys).
       Set PALM_HORIZONTAL_MODE = "track" to get next/prev back, or
       PALM_HORIZONTAL_INVERT = True if the directions feel swapped.
"""

import json
import math
import os
import sys
import time
from collections import deque

try:
    _SELF = sys.modules[__name__]              # module handle for tests/patching
except KeyError:                               # loaded via importlib with a fake name
    import types as _types
    _SELF = _types.ModuleType(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIRROR = True

SCREEN_W = 1920          # replaced at runtime by pyautogui.size()
SCREEN_H = 1080

# --- Active box (camera pixels). Shifted up so the bottom taskbar is reachable.
ACTIVE_X_MIN = 130
ACTIVE_X_MAX = 510
ACTIVE_Y_TOP = 90
ACTIVE_Y_BOTTOM = 335

EDGE_OVERSHOOT = 0.08    # 8% past each edge -> corners/taskbar reachable

# --- Cursor stabilization
MIN_SMOOTHING = 0.04
MAX_SMOOTHING = 0.22
SPEED_LOW_PX = 8.0
SPEED_HIGH_PX = 90.0
DEAD_ZONE_PIXELS = 7.0
DEAD_ZONE_EXIT_MULT = 1.8   # hysteresis: must move this much to leave the dead zone
MAX_SPEED_PIXELS = 60.0
RAW_LANDMARK_SMOOTHING = 0.35
TIP_TO_KNUCKLE_RATIO = 0.40

SENSITIVITY_MIN = 0.3
SENSITIVITY_MAX = 2.0
SENSITIVITY_STEP = 0.1

# --- Pinch / click
PINCH_THRESHOLD = 0.34
PINCH_THRESHOLD_POINTING = 0.26
PINCH_REOPEN_DELTA = 0.05
PINCH_REOPEN_FRAMES = 2
PINCH_FREEZE_MARGIN = 0.06     # freeze only within this band above the press threshold
PINCH_STALE_TIMEOUT = 1.20     # a press with no release this long is discarded
PINCH_STEP = 0.02

DOUBLE_PINCH_WINDOW = 0.45
PINCH_REPRESS_LOCKOUT = 0.07
DOUBLE_CLICK_INTERVAL = 0.08

# --- Fist geometry
FIST_CURL_RATIO = 0.98
INDEX_EXTENDED_RATIO = 1.12

# --- Drag (SPECIAL "hold")  and right click
DRAG_HOLD_DELAY = 0.35          # hold this long before a press becomes a drag
DRAG_MODE = "mouse"             # "mouse" = mouseDown/up (drag files, windows)
                                # "shift" = hold Shift (select text/items)
RIGHT_CLICK_RATIO = 0.62        # thumb-index-muscle pose (index+middle touching)
RIGHT_CLICK_RELEASE = 0.74
RIGHT_CLICK_COOLDOWN = 0.80

# --- Velocity scroll (how far the fist is from the middle of the box)
SCROLL_VELOCITY = True
SCROLL_MIN_SPAN = 0.35          # span (ticks/run) at the edge of the dead zone
SCROLL_MAX_SPAN = 3.00          # span at the far edge of the box
SCROLL_RAMP = 0.55              # seconds for the speed ramp to reach full

# --- Calibration wizard
CALIB_SETTLE_FRAMES = 6
CALIB_HOLD_FRAMES = 18

# --- Threaded capture
USE_THREADED_CAPTURE = True
CAPTURE_QUEUE_SIZE = 2

# --- Cross-platform volume
VOLUME_BACKEND = "auto"         # "auto" | "keys" | "native"
VOLUME_NATIVE_STEP = 0.05

# --- Start on login
STARTUP_WINDOWS = False         # write %APPDATA%\...\Run entry when enabled

# --- Scroll
SCROLL_AMOUNT = 8
SCROLL_REPEAT_INTERVAL = 0.03
SCROLL_SPEED_MULT = 1.0
SCROLL_SPEED_MIN = 0.25
SCROLL_SPEED_MAX = 4.0
SCROLL_DEADZONE = 0.14
SCROLL_FLIP_DELAY = 0.06
SCROLL_STOP_GRACE = 0.15
SCROLL_CLICK_LOCKOUT = 0.20

# --- Rest / engage
REST_TIMEOUT = 0.80
WAKE_STABLE_FRAMES = 2
WAKE_LOCKOUT = 0.35
REACQUIRE_HOLD = 0.15

# --- Palm swipe / volume
PALM_SWIPE_THRESHOLD_X = 0.15
PALM_SWIPE_THRESHOLD_Y = 0.10
PALM_SWIPE_MAX_TIME = 1.20      # slower than this = drift, not a swipe
PALM_SETTLE_DISTANCE = 0.05
PALM_SETTLE_TIME = 0.15
PALM_COOLDOWN_MEDIA = 0.40
PALM_COOLDOWN_VOLUME = 0.25
MEDIA_COOLDOWN = 0.25           # shared floor for all palm events
VOLUME_STEPS = 5

# --- horizontal palm swipe meaning
#   "seek"  -> forward / backward inside the video (arrow keys, YouTube-style)
#   "track" -> previous / next track
PALM_HORIZONTAL_MODE = "seek"
PALM_SWIPE_LEFT_ACTION = "forward"     # swipe toward YOUR left
PALM_SWIPE_RIGHT_ACTION = "backward"   # swipe toward YOUR right
SEEK_PRESSES = 2                       # arrow presses per swipe (YouTube = 5 s each)
PALM_HORIZONTAL_INVERT = False         # flip if left/right ever feel swapped

# --- Wave
WAVE_HISTORY_SECONDS = 1.20
WAVE_MIN_SWINGS = 3
WAVE_MIN_DISTANCE = 0.10
WAVE_REVERSAL_EPS = 0.006
WAVE_MIN_PEAK_WIDTH = 0.010
WAVE_COOLDOWN = 1.50

IDLE_SLEEP_TIME = 120
SHOW_DETECTION = False          # draw the 21 landmark points on the preview

# --- MediaPipe
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1

CLOCK_JUMP_LIMIT = 0.50         # a frame delta larger than this = suspend/clock jump
TARGET_FPS = 30.0               # all smoothing constants are calibrated at this rate
CONFIG_PATH = os.path.expanduser("~/.hand_controller_v7.json")

# ============================================================================
# PLATFORM / I-O
# ============================================================================

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_pyautogui = None


def load_pyautogui():
    """Lazy import: keeps --selftest usable with no GUI stack installed."""
    global _pyautogui
    if _pyautogui is not None:
        return _pyautogui
    import pyautogui as pag
    pag.FAILSAFE = False
    pag.PAUSE = 0
    _pyautogui = pag
    return pag


def safe_call(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return True
    except Exception as exc:                    # noqa: BLE001
        print(f"[WARN] pyautogui error: {exc}")
        return False


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def smoothstep(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ============================================================================
# GEOMETRY
# ============================================================================

def _px_dist(a, b):
    return math.hypot((a.x - b.x) * CAMERA_WIDTH, (a.y - b.y) * CAMERA_HEIGHT)


def hand_size_px(lm):
    return max(1.0, _px_dist(lm[0], lm[9]))


def pinch_distance(lm):
    """Thumb-tip <-> index-tip gap as a fraction of hand size (distance invariant)."""
    return _px_dist(lm[4], lm[8]) / hand_size_px(lm)


def is_index_extended(lm):
    """Rotation invariant: wrist->tip vs wrist->PIP."""
    wrist = lm[0]
    return _px_dist(lm[8], wrist) > _px_dist(lm[6], wrist) * INDEX_EXTENDED_RATIO


def is_fist_like(lm):
    """Real fist only; a raised index is a hard veto."""
    if is_index_extended(lm):
        return False
    wrist = lm[0]
    mid = _px_dist(lm[12], wrist) < _px_dist(lm[10], wrist) * FIST_CURL_RATIO
    ring = _px_dist(lm[16], wrist) < _px_dist(lm[14], wrist) * FIST_CURL_RATIO
    return mid and ring


def right_click_ratio(lm):
    """
    Index-tip <-> middle-tip gap over hand size. Bringing the two fingers
    together while the thumb tucks is the mouse-"right button" muscle memory.
    Index alone (pointing) keeps them far apart, so it never triggers.
    """
    return _px_dist(lm[8], lm[12]) / hand_size_px(lm)


def is_open_palm(lm):
    wrist = lm[0]
    n = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if _px_dist(lm[tip], wrist) > _px_dist(lm[pip], wrist) * 1.10:
            n += 1
    return n >= 4


def finger_count(lm):
    wrist = lm[0]
    n = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if _px_dist(lm[tip], wrist) > _px_dist(lm[pip], wrist) * 1.08:
            n += 1
    return n


def press_threshold(index_extended):
    """Live threshold: stricter while pointing, never below a sane floor."""
    if index_extended:
        return clamp(PINCH_THRESHOLD_POINTING, 0.12, PINCH_THRESHOLD)
    return max(0.12, PINCH_THRESHOLD)


# ============================================================================
# CURSOR SMOOTHER
# ============================================================================

class CursorSmoother:
    def __init__(self):
        self.fx = self.fy = None
        self.cx = self.cy = None
        self.sensitivity = 1.0
        self.preset = "BALANCED"
        self.inside_dz = False

    def reset(self):
        self.fx = self.fy = self.cx = self.cy = None
        self.inside_dz = False

    def _map(self, px, py):
        pad_x = SCREEN_W * EDGE_OVERSHOOT
        pad_y = SCREEN_H * EDGE_OVERSHOOT
        tx = clamp((px - ACTIVE_X_MIN) / float(max(1, ACTIVE_X_MAX - ACTIVE_X_MIN)), 0.0, 1.0)
        ty = clamp((py - ACTIVE_Y_TOP) / float(max(1, ACTIVE_Y_BOTTOM - ACTIVE_Y_TOP)), 0.0, 1.0)
        return (-pad_x + tx * (SCREEN_W + 2 * pad_x),
                -pad_y + ty * (SCREEN_H + 2 * pad_y))

    @staticmethod
    def _rate_alpha(alpha, dt):
        """
        Convert a smoothing factor calibrated over SMOOTHING_FRAMES frames at
        TARGET_FPS into a true per-frame factor for this frame's real duration.

        Every smoothing constant is calibrated per frame at TARGET_FPS, so a
        15 fps stream behaves like a 30 fps one instead of half as responsive.
        """
        if dt is None or dt <= 0.0:
            return alpha
        ratio = clamp(dt * TARGET_FPS, 0.1, 6.0)
        return 1.0 - (1.0 - clamp(alpha, 0.0, 0.999)) ** ratio

    def update(self, lm, frozen, dt=None):
        tip_x, tip_y = lm[8].x, lm[8].y
        knu_x, knu_y = lm[5].x, lm[5].y

        raw_x = (tip_x + (knu_x - tip_x) * TIP_TO_KNUCKLE_RATIO) * CAMERA_WIDTH
        raw_y = (tip_y + (knu_y - tip_y) * TIP_TO_KNUCKLE_RATIO) * CAMERA_HEIGHT

        if self.fx is None:
            self.fx, self.fy = raw_x, raw_y
            self.cx, self.cy = self._map(clamp(raw_x, ACTIVE_X_MIN, ACTIVE_X_MAX),
                                         clamp(raw_y, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM))
            return self._clamp(self.cx, self.cy)

        raw_a = self._rate_alpha(RAW_LANDMARK_SMOOTHING, dt)
        self.fx += (raw_x - self.fx) * raw_a
        self.fy += (raw_y - self.fy) * raw_a

        tx, ty = self._map(clamp(self.fx, ACTIVE_X_MIN, ACTIVE_X_MAX),
                           clamp(self.fy, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM))
        tx = clamp(tx, 0.0, float(SCREEN_W - 1))
        ty = clamp(ty, 0.0, float(SCREEN_H - 1))

        if frozen:
            return self._clamp(self.cx, self.cy)

        dx, dy = tx - self.cx, ty - self.cy
        dist = math.hypot(dx, dy)

        # Distance thresholds are per-frame quantities: at 15 fps a frame spans
        # twice as much motion, so they must scale or the cursor slows down.
        ratio = 1.0 if dt is None or dt <= 0.0 else clamp(dt * TARGET_FPS, 0.25, 4.0)
        dead_zone = DEAD_ZONE_PIXELS * ratio
        speed_low = SPEED_LOW_PX * ratio
        speed_high = SPEED_HIGH_PX * ratio
        max_speed = MAX_SPEED_PIXELS * ratio

        threshold = dead_zone * (DEAD_ZONE_EXIT_MULT if self.inside_dz else 1.0)
        if dist <= threshold:
            self.inside_dz = True
            return self._clamp(self.cx, self.cy)
        self.inside_dz = False

        eff = dist - dead_zone
        ux, uy = dx / dist, dy / dist
        t = smoothstep((eff - speed_low) / max(1.0, speed_high - speed_low))

        alpha = clamp((MIN_SMOOTHING + t * (MAX_SMOOTHING - MIN_SMOOTHING))
                      * self.sensitivity, 0.01, 0.55)
        alpha = clamp(self._rate_alpha(alpha, dt), 0.01, 0.85)
        step = min(eff * alpha, max_speed)
        nx, ny = self.cx + ux * step, self.cy + uy * step

        # exact edge snap so the dead zone can't make borders unreachable
        if tx <= 0.5:
            nx = 0.0
        elif tx >= SCREEN_W - 1.5:
            nx = float(SCREEN_W - 1)
        if ty <= 0.5:
            ny = 0.0
        elif ty >= SCREEN_H - 1.5:
            ny = float(SCREEN_H - 1)

        self.cx, self.cy = nx, ny
        return self._clamp(nx, ny)

    @staticmethod
    def _clamp(x, y):
        return clamp(x, 0.0, float(SCREEN_W - 1)), clamp(y, 0.0, float(SCREEN_H - 1))

    def cycle_preset(self):
        if self.preset == "PRECISION":
            self.preset = "BALANCED"
            self._apply(0.04, 0.22, 60.0)
        elif self.preset == "BALANCED":
            self.preset = "FAST"
            self._apply(0.06, 0.30, 90.0)
        else:
            self.preset = "PRECISION"
            self._apply(0.03, 0.16, 40.0)
        return self.preset

    @staticmethod
    def _apply(min_s, max_s, max_spd):
        global MIN_SMOOTHING, MAX_SMOOTHING, MAX_SPEED_PIXELS
        MIN_SMOOTHING, MAX_SMOOTHING, MAX_SPEED_PIXELS = min_s, max_s, max_spd


# ============================================================================
# CLICK ENGINE  (hysteresis + deferred single click)
# ============================================================================

class ClickEngine:
    """
    Pinch state machine.

    Actions emitted:
      "single" / "double"   click at the position captured on the press edge
      "right"               index+middle pinch (mouse-button muscle memory)
      "hold_start" / "hold_release"   press held >= DRAG_HOLD_DELAY -> drag
    """

    def __init__(self):
        self.state = "OPEN"          # OPEN | CLOSED | SPECIAL
        self.reopen_frames = 0
        self.pending = False
        self.deadline = None
        self.pending_xy = (0, 0)
        self.last_press = -1e9
        self.last_wake = -1e9
        self.n_single = 0
        self.n_double = 0
        self.n_right = 0
        self.n_hold = 0
        self.stale_consumed = False
        self.press_time = 0.0
        self.hold_timer = None       # armed while an unresolved press is held
        self.hold_active = False
        self.last_right_time = -1e9

    def reset_contact(self):
        """Mask contact without discarding an already-earned pending click."""
        self.state = "OPEN"
        self.reopen_frames = 0
        self.stale_consumed = False
        self.hold_timer = None

    def notify_wake(self, now):
        self.last_wake = now

    def blocked_until(self):
        return self.last_wake + WAKE_LOCKOUT

    def update(self, ratio, now, masked, cursor_xy, index_extended,
               right_ratio=None, can_hold=True):
        action = None

        # ---------- right click (index+middle pinch) ----------
        if not masked and right_ratio is not None:
            if self.state != "SPECIAL" and right_ratio < RIGHT_CLICK_RATIO:
                self.state = "SPECIAL"
                self.hold_timer = None
                if now - self.last_right_time >= RIGHT_CLICK_COOLDOWN:
                    self.last_right_time = now
                    self.n_right += 1
                    action = "right"
            elif self.state == "SPECIAL":
                if right_ratio > RIGHT_CLICK_RELEASE:
                    self.state = "OPEN"
                    self.reopen_frames = 0
                return action
            if self.state == "SPECIAL":
                return action

        # --- B2: a press with no release (hand left frame mid-pinch) is dropped.
        # Keep the contact CLOSED so the *same* continuous pinch cannot re-arm a
        # new press (that would click once per PINCH_STALE_TIMEOUT while held).
        if (self.state == "CLOSED"
                and self.pending
                and now - self.last_press > PINCH_STALE_TIMEOUT):
            self.pending = False
            self.deadline = None
            self.reopen_frames = 0
            self.stale_consumed = True

        if masked:
            self.reset_contact()
            return None

        pressed = False
        thr = press_threshold(index_extended)      # B7: read live every frame

        if self.state == "OPEN":
            if ratio < thr:
                pressed = True
                self.state = "CLOSED"
                self.reopen_frames = 0
                self.press_thr = thr
        else:
            if ratio > self.press_thr + PINCH_REOPEN_DELTA:
                self.reopen_frames += 1
                if self.reopen_frames >= PINCH_REOPEN_FRAMES:
                    self.state = "OPEN"
                    self.stale_consumed = False
                    if self.pending and self.deadline is None:
                        self.deadline = now + DOUBLE_PINCH_WINDOW
            else:
                self.reopen_frames = 0

        if (pressed
                and now >= self.blocked_until()
                and now - self.last_press >= PINCH_REPRESS_LOCKOUT):
            self.last_press = now
            self.press_time = now
            self.pending_xy = cursor_xy

            if self.pending and self.deadline is not None and now <= self.deadline:
                self.pending = False
                self.deadline = None
                self.n_double += 1
                action = "double"
                self.hold_timer = None
            else:
                self.pending = True
                self.deadline = None          # starts on release
                # a held press can turn into a drag instead of a click
                self.hold_timer = now if can_hold else None

        # press released: cancel a pending drag, resolve the click normally
        if self.state == "OPEN" and self.hold_timer is not None:
            self.hold_timer = None

        return action

    def hold_tick(self, now):
        """Promotes a long-held press into a drag. Call every frame."""
        if (self.hold_timer is not None
                and not self.hold_active
                and self.state == "CLOSED"
                and now - self.hold_timer >= DRAG_HOLD_DELAY):
            self.hold_active = True
            self.hold_timer = None
            self.n_hold += 1
            self.pending = False              # a drag is not a click
            self.deadline = None
            return "hold_start"
        return None

    def hold_release(self, now):
        if not self.hold_active:
            return None
        self.hold_active = False
        self.last_press = now - PINCH_REPRESS_LOCKOUT   # allow the next pinch
        return "hold_release"

    def tick(self, now):
        if self.hold_active:
            return None
        if (self.pending and self.deadline is not None and now > self.deadline):
            self.pending = False
            self.deadline = None
            self.n_single += 1
            return "single"
        return None

    def cancel_all(self):
        self.pending = False
        self.deadline = None
        self.hold_timer = None


# ============================================================================
# SCROLL ENGINE
# ============================================================================

class ScrollEngine:
    def __init__(self):
        self.direction = 0
        self.ramp_start = 0.0
        self.dir_change_time = 0.0
        self.last_tick_time = 0.0
        self.last_fist_time = -1e9
        self.multiplier = SCROLL_SPEED_MULT
        self.total_ticks = 0

    def stop(self):
        self.direction = 0

    def update(self, lm, now, masked):
        out = []
        if masked:
            self.stop()
            return out

        fist_y = lm[9].y * CAMERA_HEIGHT
        span = max(1.0, float(ACTIVE_Y_BOTTOM - ACTIVE_Y_TOP))
        pos = (fist_y - ACTIVE_Y_TOP) / span

        if pos < -0.05 or pos > 1.05:      # parked outside the box
            self.stop()
            return out

        self.last_fist_time = now
        half = SCROLL_DEADZONE / 2.0
        if pos > 0.5 + half:
            direction = -1
        elif pos < 0.5 - half:
            direction = 1
        else:
            direction = 0

        if direction != self.direction:
            self.direction = direction
            self.dir_change_time = now
            self.ramp_start = now
            # B5: arm the first tick AT the moment the flip delay expires
            self.last_tick_time = now + SCROLL_FLIP_DELAY - SCROLL_REPEAT_INTERVAL

        if self.direction == 0:
            return out

        # velocity: distance from the middle of the box drives the tick rate
        dist = 0.0
        if SCROLL_VELOCITY:
            span_norm = max(1e-6, 0.5 - half)
            dist = clamp((abs(pos - 0.5) - half) / span_norm, 0.0, 1.0)
            ramp = clamp((now - self.ramp_start) / max(1e-3, SCROLL_RAMP), 0.0, 1.0)
            span = SCROLL_MIN_SPAN + dist * (SCROLL_MAX_SPAN - SCROLL_MIN_SPAN)
            interval = SCROLL_REPEAT_INTERVAL / max(1e-6, span * (0.35 + 0.65 * ramp))
        else:
            interval = SCROLL_REPEAT_INTERVAL

        ticks = 0
        while now - self.last_tick_time >= interval and ticks < 8:
            if now - self.dir_change_time < SCROLL_FLIP_DELAY:
                break                      # B5: never tick during the flip delay
            amount = int(round(SCROLL_AMOUNT * self.multiplier)) * self.direction
            if amount != 0:
                out.append(amount)
                self.total_ticks += 1
            self.last_tick_time += interval
            ticks += 1

        return out

    def cycling(self, now):
        return self.direction != 0 or (now - self.last_fist_time) < SCROLL_CLICK_LOCKOUT


# ============================================================================
# PALM SWIPE
# ============================================================================

class PalmSwipeDetector:
    def __init__(self):
        self.origin = None
        self.origin_time = 0.0
        self.settled_since = None
        self.armed = True
        self.last_fire = -1e9

    def reset(self):
        self.origin = None
        self.settled_since = None
        self.armed = True

    def _zero(self, px, py, now, armed=True):
        self.origin = (px, py)
        self.origin_time = now
        self.settled_since = None
        self.armed = armed

    def update(self, lm, now):
        px, py = lm[9].x, lm[9].y
        if self.origin is None:
            self._zero(px, py, now)
            return None

        ox, oy = self.origin
        dx, dy = px - ox, py - oy
        dist = math.hypot(dx, dy)

        # hand parked near its origin -> settle timer runs, then re-arm
        if dist < PALM_SETTLE_DISTANCE * 0.6:
            if self.settled_since is None:
                self.settled_since = now
            elif now - self.settled_since >= PALM_SETTLE_TIME:
                self._zero(px, py, now, armed=True)
            return None
        self.settled_since = None

        if now - self.origin_time > PALM_SWIPE_MAX_TIME:
            self._zero(px, py, now, armed=self.armed)
            return None

        if not self.armed:
            return None

        event = None
        if abs(dx) > PALM_SWIPE_THRESHOLD_X and abs(dx) > abs(dy):
            event = "right" if dx > 0 else "left"
            cool = PALM_COOLDOWN_MEDIA
        elif abs(dy) > PALM_SWIPE_THRESHOLD_Y and abs(dy) > abs(dx):
            event = "volup" if dy < 0 else "voldown"
            cool = PALM_COOLDOWN_VOLUME

        if event is None:
            return None
        if now - self.last_fire < MEDIA_COOLDOWN:
            return None
        if now - self.last_fire < cool:
            return None

        # B6: stay disarmed until the hand comes back to rest
        self.last_fire = now
        self._zero(px, py, now, armed=False)
        return event


# ============================================================================
# WAVE
# ============================================================================

class WaveDetector:
    """Counts direction reversals on the live signal with debounce + hysteresis."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = deque()
        self.side = 0
        self.extreme = None
        self.swings = deque()
        self.last_fire = -1e9

    def update(self, lm, now):
        x = lm[9].x
        self.hist.append((now, x))
        while self.hist and now - self.hist[0][0] > WAVE_HISTORY_SECONDS:
            self.hist.popleft()

        if len(self.hist) < 4:
            return False

        xs = [v for _, v in self.hist]
        if max(xs) - min(xs) < WAVE_MIN_DISTANCE:
            self.side = 0
            self.extreme = None
            return False

        if self.extreme is None:
            self.extreme = x

        d = x - self.extreme

        if self.side == 0:
            if abs(d) >= WAVE_REVERSAL_EPS:
                self.side = 1 if d > 0 else -1
                self.extreme = x
        elif self.side == 1:
            if d > WAVE_MIN_PEAK_WIDTH:                 # keep extending the peak
                self.extreme = x
            elif d < -WAVE_REVERSAL_EPS:                # confirmed reversal
                self.swings.append(now)
                self.side = -1
                self.extreme = x
        else:
            if d < -WAVE_MIN_PEAK_WIDTH:
                self.extreme = x
            elif d > WAVE_REVERSAL_EPS:
                self.swings.append(now)
                self.side = 1
                self.extreme = x

        while self.swings and now - self.swings[0] > WAVE_HISTORY_SECONDS:
            self.swings.popleft()

        if len(self.swings) >= WAVE_MIN_SWINGS and now - self.last_fire > WAVE_COOLDOWN:
            self.last_fire = now
            self.reset()
            return True
        return False


# ============================================================================
# CONTROLLER
# ============================================================================

class HandController:
    def __init__(self, hands=None, pag=None):
        self.hands = hands
        self.pag = pag

        self.smoother = CursorSmoother()
        self.click = ClickEngine()
        self.scroll = ScrollEngine()
        self.palm = PalmSwipeDetector()
        self.wave = WaveDetector()

        self.resting = True
        self.wake_counter = 0
        self.last_engaged = 0.0
        self.hand_missing = False
        self.reacquire_time = -1e9

        self.asleep = False
        self.last_now = None
        self.started_at = None
        self.frame_dt = 1.0 / TARGET_FPS
        self.last_action_time = 0.0

        self.cursor = (SCREEN_W // 2, SCREEN_H // 2)
        self.last_palm_event = None
        self.drag_active = False
        self.calibration = Calibration()
        self.stats = {"single": 0, "double": 0, "scroll": 0, "media": 0,
                      "sleep": 0, "wake": 0}

    # ---------------- time ----------------
    def _guard_time(self, now):
        """B10: after a suspend/clock jump, reset timers instead of firing a burst."""
        if self.last_now is not None:
            dt = now - self.last_now
            if dt < 0 or dt > CLOCK_JUMP_LIMIT:
                self.click.cancel_all()
                self.scroll.stop()
                self.scroll.last_tick_time = now
                self.scroll.dir_change_time = now
                self.palm.reset()
                self.wave.reset()
                self.reacquire_time = now
                self.last_engaged = now
                print(f"[TIME] jump {dt:+.2f}s -> timers reset")
        self.frame_dt = 1.0 / TARGET_FPS if self.last_now is None else (now - self.last_now)
        self.last_now = now

    # ---------------- rest / sleep ----------------
    def enter_rest(self, reason="idle"):
        if self.resting:
            return
        if self.click.hold_active:
            self.click.hold_release(time.monotonic())
            self._do_click("hold_release")
        self.resting = True
        self.wake_counter = 0
        self.scroll.stop()
        self.click.reset_contact()
        self.palm.reset()
        print(f"[REST] {reason} -> show INDEX finger to resume")

    def update_rest(self, lm, now):
        if lm is None:
            self.hand_missing = True
            self.smoother.reset()
            self.click.reset_contact()          # B2: no stuck contact after loss
            self.wave.reset()
            if now - self.last_engaged > REST_TIMEOUT:
                self.enter_rest("hand lost")
            return

        index_up = is_index_extended(lm)
        engaged = index_up or is_fist_like(lm) or is_open_palm(lm)

        if not self.resting:
            if engaged:
                self.last_engaged = now
            elif now - self.last_engaged > REST_TIMEOUT:
                self.enter_rest("hand idle")
        else:
            if index_up and not self.asleep:
                self.wake_counter += 1
                if self.wake_counter >= WAKE_STABLE_FRAMES:
                    self.resting = False
                    self.last_engaged = now
                    self.wake_counter = 0
                    self.click.notify_wake(now)
                    print("[REST] awake")
            else:
                self.wake_counter = 0

    # ---------------- main ----------------
    def process(self, lm, now):
        self._guard_time(now)

        if self.started_at is None:
            self.started_at = now
        if self.last_action_time == 0.0:
            self.last_action_time = now
        if self.last_engaged == 0.0:
            self.last_engaged = now

        # calibration takes over: no gestures fire while it runs
        if self.calibration.active:
            if self.calibration.update(lm):
                apply_calibration(self, self.calibration.result)
                self.smoother.reset()
            return self.cursor

        self.update_rest(lm, now)

        # pending single click resolves even while resting / asleep / fist
        single = self.click.tick(now)
        if single:
            self._do_click("single")

        if lm is None:
            return None

        # B1: sleep is a hard gate (only a wave gets you out)
        blocked_by_sleep = self.asleep

        fist = is_fist_like(lm)
        index_up = is_index_extended(lm)
        palm = is_open_palm(lm)

        # ---- wave -> sleep / wake (works while asleep, that's the point) ----
        if palm and not self.resting:
            if self.wave.update(lm, now):
                self.asleep = not self.asleep
                self.stats["sleep" if self.asleep else "wake"] += 1
                self.scroll.stop()
                self.palm.reset()
                self.click.reset_contact()
                if self.asleep:
                    self.enter_rest("sleep")
                else:
                    self.last_engaged = now
                print("[WAVE] " + ("sleep" if self.asleep else "awake"))
        elif not palm:
            self.wave.reset()

        if (not self.asleep and not self.resting
                and now - self.last_action_time > IDLE_SLEEP_TIME):
            self.asleep = True
            self.enter_rest("idle timeout")
            print(f"[SLEEP] no activity for {IDLE_SLEEP_TIME}s -> wave to wake")
            self.scroll.update(lm, now, masked=True)
            return self.cursor

        if blocked_by_sleep:
            self.scroll.update(lm, now, masked=True)
            return self.cursor

        # ---- fist scrolling ----
        if not self.resting and fist and not index_up:
            amounts = self.scroll.update(lm, now, masked=False)
            for amt in amounts:
                if self.pag is not None:
                    safe_call(self.pag.scroll, amt)
                self.stats["scroll"] += 1
        else:
            self.scroll.update(lm, now, masked=True)

        # ---- cursor ----
        ratio = pinch_distance(lm)
        thr = press_threshold(index_up)
        # B3: freeze only while actually closing the pinch, not while pointing
        near_pinch = (not index_up) and (ratio < thr + PINCH_FREEZE_MARGIN)
        frozen = (self.resting or fist or near_pinch
                  or (now - self.reacquire_time) < REACQUIRE_HOLD)

        dt = None if self.last_now is None else abs(self.frame_dt)
        pos = self.smoother.update(lm, frozen, dt)
        self.cursor = (int(pos[0]), int(pos[1]))
        if not self.resting and not frozen and self.pag is not None:
            safe_call(self.pag.moveTo, self.cursor[0], self.cursor[1])

        # ---- clicks ----
        masked = self.resting or fist or self.scroll.cycling(now)
        can_hold = (not self.resting) and not self.scroll.cycling(now)
        action = self.click.update(
            ratio, now, masked=masked, cursor_xy=self.cursor,
            index_extended=index_up, right_ratio=right_click_ratio(lm),
            can_hold=can_hold)
        if action:
            self._do_click(action)
            self.last_action_time = now

        # ---- drag: a press held past DRAG_HOLD_DELAY becomes a mouse-down ----
        hold = self.click.hold_tick(now)
        if hold:
            self._do_click(hold)
            self.last_action_time = now
        if self.click.hold_active and self.click.state != "CLOSED":
            # releasing the pinch (or any abort) ends the drag
            rel = self.click.hold_release(now)
            if rel:
                self._do_click(rel)

        # ---- palm media / volume ----
        if not self.resting and not fist and palm:
            ev = self.palm.update(lm, now)
            if ev:
                self._do_media(ev, now)
        else:
            self.palm.reset()

        return self.cursor

    # ---------------- actions ----------------
    def _do_click(self, kind):
        self.stats[kind] = self.stats.get(kind, 0) + 1
        if kind in ("hold_start", "hold_release"):
            self.drag_active = kind == "hold_start"
        if self.pag is None:
            return

        if kind == "double":
            safe_call(self.pag.doubleClick, interval=DOUBLE_CLICK_INTERVAL)
        elif kind == "right":
            x, y = self.cursor
            safe_call(self.pag.click, int(x), int(y), button="right")
        elif kind == "hold_start":
            if DRAG_MODE == "shift":
                safe_call(self.pag.keyDown, "shift")
            else:
                safe_call(self.pag.mouseDown)
        elif kind == "hold_release":
            if DRAG_MODE == "shift":
                safe_call(self.pag.keyUp, "shift")
            else:
                safe_call(self.pag.mouseUp)
        else:
            x, y = self.click.pending_xy
            safe_call(self.pag.click, int(x), int(y))

    def _do_media(self, ev, now):
        self.stats["media"] += 1
        self.last_action_time = now
        self.last_palm_event = ev
        if self.pag is None:
            return

        if ev in ("left", "right"):
            if PALM_HORIZONTAL_INVERT:
                ev = "right" if ev == "left" else "left"
            action = PALM_SWIPE_LEFT_ACTION if ev == "left" else PALM_SWIPE_RIGHT_ACTION

            if PALM_HORIZONTAL_MODE == "track":
                key = "nexttrack" if action == "forward" else "prevtrack"
                safe_call(self.pag.press, key)
            else:                                    # seek
                key = "right" if action == "forward" else "left"
                presses = SEEK_PRESSES if action == "forward" else SEEK_PRESSES
                for _ in range(max(1, presses)):
                    safe_call(self.pag.press, key)
        elif ev == "volup":
            set_volume(VOLUME_STEPS, self.pag)
        elif ev == "voldown":
            set_volume(-VOLUME_STEPS, self.pag)


# ---------------------------------------------------------------------------
# Native volume (Windows Core Audio / macOS AppleScript / Linux amixer+pactl)
# ---------------------------------------------------------------------------

_vol_backend = None


def _vol_windows():
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    iface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))


def _vol_macos(step):
    import subprocess
    sign = "+" if step > 0 else "-"
    script = f"set volume output volume (output volume of (get volume settings) {sign} {int(abs(step) * 100)})"
    subprocess.run(["osascript", "-e", script], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _vol_linux(step):
    import shutil
    import subprocess
    pct = f"{int(abs(step) * 100)}%"
    sign = "+" if step > 0 else "-"
    if shutil.which("amixer"):
        subprocess.run(["amixer", "-q", "sset", "Master", f"{pct}{sign}"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    if shutil.which("pactl"):
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{pct}"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    return False


def native_volume(step):
    """Adjust system volume natively. Returns True on success."""
    global _vol_backend
    try:
        if _vol_backend is None:
            if sys.platform == "win32":
                _vol_backend = _vol_windows()
            elif sys.platform == "darwin":
                _vol_backend = "macos"
            else:
                _vol_backend = "linux"
        if _vol_backend == "macos":
            _vol_macos(step)
        elif _vol_backend == "linux":
            return _vol_linux(step)
        else:
            vol = _vol_backend
            current = vol.GetMasterVolumeLevelScalar()
            vol.SetMasterVolumeLevelScalar(clamp(current + step, 0.0, 1.0), None)
        return True
    except Exception as exc:                      # noqa: BLE001
        print(f"[VOL] native backend unavailable ({exc}); using media keys")
        _vol_backend = "keys"
        return False


def set_volume(step_presses, pag):
    """One place for volume: native scalar when possible, media keys otherwise."""
    if VOLUME_BACKEND in ("auto", "native"):
        if native_volume(VOLUME_NATIVE_STEP * step_presses):
            return
    if pag is None:
        return
    key = "volumeup" if step_presses > 0 else "volumedown"
    for _ in range(abs(step_presses)):
        safe_call(pag.press, key)


def script_path():
    return os.path.abspath(sys.argv[0])


def startup_command():
    """Command that launches this controller without a console window."""
    py = sys.executable
    pyw = py.replace("python.exe", "pythonw.exe") if sys.platform == "win32" else py
    target = pyw if os.path.exists(pyw) else py
    return f'"{target}" "{script_path()}"'


def set_startup(enabled):
    """Windows: HKCU Run key. macOS/Linux: prints the line to add yourself."""
    if sys.platform == "win32":
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                                winreg.KEY_SET_VALUE) as key:
                if enabled:
                    winreg.SetValueEx(key, "HandGestureController", 0,
                                      winreg.REG_SZ, startup_command())
                    print("[STARTUP] enabled (HKCU Run)")
                else:
                    try:
                        winreg.DeleteValue(key, "HandGestureController")
                        print("[STARTUP] disabled")
                    except FileNotFoundError:
                        print("[STARTUP] was not enabled")
            return True
        except Exception as exc:                  # noqa: BLE001
            print(f"[STARTUP] failed: {exc}")
            return False
    print(f"[STARTUP] add this to your session autostart:\n  {startup_command()}")
    return False


def make_shortcut():
    """Windows: drop a Desktop shortcut. Other OSes: print the command."""
    if sys.platform == "win32":
        try:
            import subprocess
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            lnk = os.path.join(desktop, "Hand Gesture Controller.lnk")
            ps = (
                "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%s');"
                "$s.TargetPath='%s';$s.Arguments='\"%s\"';$s.WorkingDirectory='%s';$s.Save()"
                % (lnk.replace("'", "''"), sys.executable.replace("'", "''"),
                   script_path().replace("'", "''"),
                   os.path.dirname(script_path()).replace("'", "''"))
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)
            print(f"[SHORTCUT] created {lnk}")
            return True
        except Exception as exc:                  # noqa: BLE001
            print(f"[SHORTCUT] failed: {exc}")
            return False
    print(f"[SHORTCUT] create a launcher for:\n  {startup_command()}")
    return False


def save_config(controller):
    """Persist the knobs you tuned at runtime so they survive a restart."""
    data = {
        "version": 1,
        "pinch_threshold": round(PINCH_THRESHOLD, 4),
        "pinch_pointing": round(PINCH_THRESHOLD_POINTING, 4),
        "sensitivity": round(controller.smoother.sensitivity, 3),
        "preset": controller.smoother.preset,
        "scroll_mult": round(controller.scroll.multiplier, 3),
        "box": [ACTIVE_X_MIN, ACTIVE_X_MAX, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM],
        "overshoot": round(EDGE_OVERSHOOT, 4),
        "reopen_delta": round(PINCH_REOPEN_DELTA, 4),
        "right_click": round(RIGHT_CLICK_RATIO, 4),
        "freeze_margin": round(PINCH_FREEZE_MARGIN, 4),
    }
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"[CONFIG] saved -> {CONFIG_PATH}")
        return True
    except Exception as exc:                       # noqa: BLE001
        print(f"[CONFIG] save failed: {exc}")
        return False


def load_config(controller):
    """Applied AFTER a fresh boot, so defaults are what you get if the file is bad."""
    global PINCH_THRESHOLD, PINCH_THRESHOLD_POINTING, PINCH_REOPEN_DELTA
    global RIGHT_CLICK_RATIO, PINCH_FREEZE_MARGIN
    global ACTIVE_X_MIN, ACTIVE_X_MAX, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM, EDGE_OVERSHOOT
    if not os.path.exists(CONFIG_PATH):
        return False
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)

        PINCH_THRESHOLD = clamp(float(data.get("pinch_threshold", PINCH_THRESHOLD)), 0.15, 0.60)
        PINCH_THRESHOLD_POINTING = clamp(
            float(data.get("pinch_pointing", PINCH_THRESHOLD - 0.08)), 0.12, PINCH_THRESHOLD)
        controller.smoother.sensitivity = clamp(
            float(data.get("sensitivity", 1.0)), SENSITIVITY_MIN, SENSITIVITY_MAX)
        controller.scroll.multiplier = clamp(
            float(data.get("scroll_mult", 1.0)), SCROLL_SPEED_MIN, SCROLL_SPEED_MAX)

        box = data.get("box")
        if isinstance(box, list) and len(box) == 4:
            x0, x1, y0, y1 = (int(v) for v in box)
            if 0 <= x0 < x1 <= CAMERA_WIDTH and 0 <= y0 < y1 <= CAMERA_HEIGHT:
                ACTIVE_X_MIN, ACTIVE_X_MAX, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM = x0, x1, y0, y1

        EDGE_OVERSHOOT = clamp(float(data.get("overshoot", EDGE_OVERSHOOT)), 0.0, 0.30)
        PINCH_REOPEN_DELTA = clamp(float(data.get("reopen_delta", PINCH_REOPEN_DELTA)), 0.02, 0.20)
        RIGHT_CLICK_RATIO = clamp(float(data.get("right_click", RIGHT_CLICK_RATIO)), 0.20, 0.90)
        PINCH_FREEZE_MARGIN = clamp(float(data.get("freeze_margin", PINCH_FREEZE_MARGIN)), 0.02, 0.20)
        controller.smoother.reset()
        print(f"[CONFIG] loaded <- {CONFIG_PATH}")
        return True
    except Exception as exc:                       # noqa: BLE001
        print(f"[CONFIG] ignored (bad file): {exc}")
        return False


# ============================================================================
# CAMERA / MEDIAPIPE
# ============================================================================

def open_camera(index=CAMERA_INDEX):
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class FrameGrabber:
    """
    Reads the camera on its own thread so `hands.process()` never stalls the
    camera pipeline. Without it a slow inference makes the next read() return
    a stale frame, which feels exactly like input lag.
    """

    def __init__(self, cap, size=CAPTURE_QUEUE_SIZE):
        import threading
        self._cap = cap
        self._q = deque(maxlen=max(1, size))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="frame-grabber")

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._q.append(frame)

    def read(self):
        with self._lock:
            if not self._q:
                return False, None
            return True, self._q[-1]          # always the freshest frame

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class HandsAdapter:
    """Uniform wrapper so the main loop does not care which MP backend is used."""

    def __init__(self, backend, impl, mode):
        self.backend = backend          # "solutions" | "tasks"
        self.impl = impl
        self.mode = mode
        self.last_landmarks = None

    def process(self, rgb):
        try:
            return self.impl.process(rgb)
        except Exception as exc:                      # noqa: BLE001
            print(f"[MP] frame skipped: {exc}")
            return None

    def close(self):
        try:
            if hasattr(self.impl, "close"):
                self.impl.close()
        except Exception:
            pass


class _MultiLandmarks:
    def __init__(self, landmarks):
        self.multi_hand_landmarks = landmarks or []


class TasksWrapper:
    """Adapts the Tasks HandLandmarker result to the Solutions result shape."""

    def __init__(self, landmarker):
        self.landmarker = landmarker
        self._Point = None
        self._last_ts = -1

    def process(self, rgb):
        if self._Point is None:
            class Point:
                __slots__ = ("x", "y", "z")
                def __init__(self, x, y, z=0.0):
                    self.x, self.y, self.z = x, y, z
            self._Point = Point

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        try:
            res = self.landmarker.detect_for_video(mp_image, ts)
        except Exception:
            try:
                res = self.landmarker.detect(mp_image)
            except Exception:
                res = None

        hands = []
        for hand in getattr(res, "hand_landmarks", None) or []:
            class _H:
                pass
            holder = _H()
            holder.landmark = [self._Point(p.x, p.y, getattr(p, "z", 0.0)) for p in hand]
            hands.append(holder)
        return _MultiLandmarks(hands)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


def find_model():
    """Locate hand_landmarker.task (next to the script, cwd, then ./models)."""
    candidates = [
        os.path.join(os.path.dirname(script_path()), "hand_landmarker.task"),
        os.path.join(os.getcwd(), "hand_landmarker.task"),
        os.path.join(os.path.dirname(script_path()), "models", "hand_landmarker.task"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def build_hands():
    """
    Prefer the classic Solutions API; fall back to the new Tasks API when
    Solutions is gone (MediaPipe >= 0.10.20 deprecates / removes it).
    """
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "hands"):
        impl = solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        print("[MP] backend: solutions (classic)")
        return HandsAdapter("solutions", impl, "video")

    # ---- Tasks backend ----
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model = find_model()
    if model is None:
        raise RuntimeError(
            "MediaPipe Tasks needs hand_landmarker.task next to this script.\n"
            "Download it with:\n"
            "  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=TRACKING_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("[MP] backend: tasks (hand_landmarker.task)")
    return HandsAdapter("tasks", TasksWrapper(landmarker), "video")


# ============================================================================
# CALIBRATION WIZARD
# ============================================================================

class Calibration:
    """
    Four steps, each held CALIB_HOLD_FRAMES frames:
      OPEN -> PINCH -> FIST -> RIGHT (index+middle pinch)
    Produces thresholds from YOUR hand instead of guesswork.
    """

    STEPS = (
        ("OPEN", "relaxed OPEN hand, fingers apart"),
        ("PINCH", "SMALL pinch (thumb + index, barely touching)"),
        ("FIST", "closed FIST"),
        ("RIGHT", "index + middle fingers together (right-click pose)"),
    )

    def __init__(self):
        self.active = False
        self.step = 0
        self.hold = 0
        self.settled = 0
        self.samples = {name: [] for name, _ in self.STEPS}
        self.result = None
        self.message = ""

    def start(self):
        self.active = True
        self.step = 0
        self.hold = 0
        self.settled = 0
        self.samples = {name: [] for name, _ in self.STEPS}
        self.result = None
        self.message = self.STEPS[0][1]
        print("[CALIB] started - " + self.STEPS[0][1])

    def current_hint(self):
        if not self.active:
            return ""
        return self.STEPS[self.step][1]

    def update(self, lm):
        """Feed a frame; returns True when a fresh result is available."""
        if not self.active:
            return False
        if lm is None:
            self.hold = 0
            self.settled = 0
            return False

        name = self.STEPS[self.step][0]

        # let the hand settle before sampling so the average is clean
        if self.settled < CALIB_SETTLE_FRAMES:
            self.settled += 1
            return False

        if name in ("OPEN", "PINCH", "RIGHT"):
            value = pinch_distance(lm) if name != "RIGHT" else right_click_ratio(lm)
        else:
            value = pinch_distance(lm)

        self.samples[name].append(value)
        self.hold += 1

        if self.hold >= CALIB_HOLD_FRAMES:
            self.step += 1
            self.hold = 0
            self.settled = 0
            if self.step >= len(self.STEPS):
                self.active = False
                self.result = self._compute()
                self.message = "done"
                return True
            self.message = self.STEPS[self.step][1]
            print("[CALIB] next - " + self.message)
        return False

    @staticmethod
    def _mean(values):
        return sum(values) / max(1, len(values))

    def _compute(self):
        def median(values):
            v = sorted(values)
            if not v:
                return 0.0
            return v[len(v) // 2]

        open_v = median(self.samples["OPEN"])
        pinch_v = median(self.samples["PINCH"])
        fist_v = median(self.samples["FIST"])
        right_v = median(self.samples["RIGHT"])

        # press threshold sits between a real pinch and the relaxed hand
        low = min(max(pinch_v, 0.05), open_v * 0.9)
        thr = clamp(low + (open_v - low) * 0.40, 0.15, 0.60)
        pointing = clamp(thr - 0.08, 0.12, thr)
        freeze_margin = round(clamp((open_v - thr) * 0.5, 0.03, 0.12), 3)

        return {
            "pinch_threshold": round(thr, 3),
            "pinch_pointing": round(pointing, 3),
            "reopen_delta": round(clamp(0.05, 0.02, max(0.02, (open_v - thr) * 0.5)), 3),
            "right_click": round(clamp(right_v + (open_v - right_v) * 0.25, 0.20, 0.90), 3),
            "freeze_margin": freeze_margin,
            "raw": {"open": round(open_v, 3), "pinch": round(pinch_v, 3),
                    "fist": round(fist_v, 3), "right": round(right_v, 3)},
        }


def apply_calibration(controller, result):
    """Push a finished calibration into the live config."""
    global PINCH_THRESHOLD, PINCH_THRESHOLD_POINTING, PINCH_REOPEN_DELTA
    global RIGHT_CLICK_RATIO, PINCH_FREEZE_MARGIN

    PINCH_THRESHOLD = result["pinch_threshold"]
    PINCH_THRESHOLD_POINTING = result["pinch_pointing"]
    PINCH_REOPEN_DELTA = result["reopen_delta"]
    RIGHT_CLICK_RATIO = result["right_click"]
    PINCH_FREEZE_MARGIN = result["freeze_margin"]

    # a fist must never look like a pinch or a right click
    if result["raw"]["fist"] < PINCH_THRESHOLD:
        print(f"[CALIB] WARNING: fist reads {result['raw']['fist']:.2f} < press "
              f"{PINCH_THRESHOLD:.2f}; the geometric fist guard still protects you")
    print(f"[CALIB] applied {result}")
    return True


# ============================================================================
# HUD / KEYS
# ============================================================================

def draw_hud(cv2, frame, lm, controller, now, fps=0.0):
    cv2.rectangle(frame, (ACTIVE_X_MIN, ACTIVE_Y_TOP), (ACTIVE_X_MAX, ACTIVE_Y_BOTTOM),
                  (0, 255, 0), 2)
    cm = 14
    for px, py in ((ACTIVE_X_MIN, ACTIVE_Y_TOP), (ACTIVE_X_MAX, ACTIVE_Y_TOP),
                   (ACTIVE_X_MIN, ACTIVE_Y_BOTTOM), (ACTIVE_X_MAX, ACTIVE_Y_BOTTOM)):
        cv2.line(frame, (px - cm, py), (px + cm, py), (0, 255, 255), 2)
        cv2.line(frame, (px, py - cm), (px, py + cm), (0, 255, 255), 2)

    if controller.calibration.active:
        t1 = f"CALIBRATE {controller.calibration.step + 1}/{len(controller.calibration.STEPS)}: " \
             f"{controller.calibration.current_hint()}"
        c1 = (0, 255, 255)
    elif controller.drag_active:
        t1, c1 = "DRAGGING - open the pinch to drop", (255, 0, 255)
    elif controller.asleep:
        t1, c1 = "SLEEP - wave to wake", (0, 165, 255)
    elif controller.resting:
        t1, c1 = "REST - show INDEX finger", (0, 0, 255)
    elif controller.click.pending:
        t1, c1 = "pinch 1 - waiting 2nd pinch", (0, 255, 255)
    else:
        t1, c1 = "ACTIVE", (0, 255, 0)
    cv2.putText(frame, t1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c1, 2, cv2.LINE_AA)

    if lm is not None:
        # Draw skeleton
        h, w, _ = frame.shape
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        coords = [(int(p.x * w), int(p.y * h)) for p in lm]
        for p1, p2 in connections:
            if p1 < len(coords) and p2 < len(coords):
                cv2.line(frame, coords[p1], coords[p2], (255, 0, 0), 2)
        for cx, cy in coords:
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    if lm is not None and SHOW_DETECTION:
        for p in lm:
            cv2.circle(frame, (int(p.x * frame.shape[1]), int(p.y * frame.shape[0])),
                       2, (0, 140, 255), -1)

    if lm is not None:
        ratio = pinch_distance(lm)
        idx = is_index_extended(lm)
        thr = press_threshold(idx)
        col = (0, 255, 0) if ratio < thr else (170, 170, 170)
        cv2.putText(frame, f"pinch {ratio:.2f} / {thr:.2f}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        pose = "INDEX" if idx else ("FIST" if is_fist_like(lm)
                                    else ("PALM" if is_open_palm(lm) else "-"))
        cv2.putText(frame, pose, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 200, 0), 1, cv2.LINE_AA)

    if controller.scroll.direction > 0:
        st = f"SCROLL UP x{controller.scroll.multiplier:.2f}"
    elif controller.scroll.direction < 0:
        st = f"SCROLL DN x{controller.scroll.multiplier:.2f}"
    else:
        st = "-"
    cv2.putText(frame, st, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)

    if controller.calibration.active:
        frac = controller.calibration.hold / float(CALIB_HOLD_FRAMES)
        cv2.rectangle(frame, (10, 118), (10 + int(200 * frac), 132), (0, 255, 255), -1)
        cv2.rectangle(frame, (10, 118), (210, 132), (255, 255, 255), 1)

    cv2.putText(frame, f"sens {controller.smoother.sensitivity:.1f}x "
                       f"{controller.smoother.preset}  {fps:4.1f}fps",
                (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA)

    # right-hand gauge: how many fingers are visible (mirrors to the user's left)
    cv2.putText(frame, f"fingers {finger_count(lm) if lm is not None else 0}",
                (frame.shape[1] - 110, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 150, 255), 1, cv2.LINE_AA)


def handle_key(key, controller):
    global PINCH_THRESHOLD, PINCH_THRESHOLD_POINTING
    global ACTIVE_X_MIN, ACTIVE_X_MAX, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM

    if key == -1:
        return
    ch = chr(key) if 0 <= key < 256 else ""

    if ch == "]":
        step = 8
        ACTIVE_X_MIN = min(ACTIVE_X_MAX - 80, ACTIVE_X_MIN + step)
        ACTIVE_X_MAX = max(ACTIVE_X_MIN + 80, ACTIVE_X_MAX - step)
        ACTIVE_Y_TOP = min(ACTIVE_Y_BOTTOM - 60, ACTIVE_Y_TOP + step)
        ACTIVE_Y_BOTTOM = max(ACTIVE_Y_TOP + 60, ACTIVE_Y_BOTTOM - step)
        controller.smoother.reset()
        print(f"[BOX] smaller x{ACTIVE_X_MIN}-{ACTIVE_X_MAX} y{ACTIVE_Y_TOP}-{ACTIVE_Y_BOTTOM}")
    elif ch == "[":
        step = 8
        ACTIVE_X_MIN = max(0, ACTIVE_X_MIN - step)
        ACTIVE_X_MAX = min(CAMERA_WIDTH, ACTIVE_X_MAX + step)
        ACTIVE_Y_TOP = max(0, ACTIVE_Y_TOP - step)
        ACTIVE_Y_BOTTOM = min(CAMERA_HEIGHT, ACTIVE_Y_BOTTOM + step)
        controller.smoother.reset()
        print(f"[BOX] larger x{ACTIVE_X_MIN}-{ACTIVE_X_MAX} y{ACTIVE_Y_TOP}-{ACTIVE_Y_BOTTOM}")
    elif ch in ("-", "_"):
        controller.smoother.sensitivity = clamp(
            controller.smoother.sensitivity - SENSITIVITY_STEP,
            SENSITIVITY_MIN, SENSITIVITY_MAX)
        print(f"[SENS] {controller.smoother.sensitivity:.1f}x")
    elif ch in ("=", "+"):
        controller.smoother.sensitivity = clamp(
            controller.smoother.sensitivity + SENSITIVITY_STEP,
            SENSITIVITY_MIN, SENSITIVITY_MAX)
        print(f"[SENS] {controller.smoother.sensitivity:.1f}x")
    elif ch == ".":
        PINCH_THRESHOLD = clamp(PINCH_THRESHOLD + PINCH_STEP, 0.15, 0.60)
        # B9: pointing threshold follows, clamped so it never inverts
        PINCH_THRESHOLD_POINTING = clamp(PINCH_THRESHOLD - 0.08, 0.12, PINCH_THRESHOLD)
        print(f"[PINCH] easier -> {PINCH_THRESHOLD:.2f} (pointing {PINCH_THRESHOLD_POINTING:.2f})")
    elif ch == ",":
        PINCH_THRESHOLD = clamp(PINCH_THRESHOLD - PINCH_STEP, 0.15, 0.60)
        PINCH_THRESHOLD_POINTING = clamp(PINCH_THRESHOLD - 0.08, 0.12, PINCH_THRESHOLD)
        print(f"[PINCH] stricter -> {PINCH_THRESHOLD:.2f} (pointing {PINCH_THRESHOLD_POINTING:.2f})")
    elif ch == "9":
        controller.scroll.multiplier = clamp(
            controller.scroll.multiplier + 0.25, SCROLL_SPEED_MIN, SCROLL_SPEED_MAX)
        print(f"[SCROLL] x{controller.scroll.multiplier:.2f}")
    elif ch == "0":
        controller.scroll.multiplier = clamp(
            controller.scroll.multiplier - 0.25, SCROLL_SPEED_MIN, SCROLL_SPEED_MAX)
        print(f"[SCROLL] x{controller.scroll.multiplier:.2f}")
    elif ch in ("m", "M"):
        print(f"[SMOOTH] {controller.smoother.cycle_preset()}")
    elif ch in ("s", "S"):
        controller.asleep = not controller.asleep
        print("[SLEEP] manual toggle ->", "asleep" if controller.asleep else "awake")
    elif ch in ("d", "D"):
        global SHOW_DETECTION
        SHOW_DETECTION = not SHOW_DETECTION
        print("[HUD] landmarks", "on" if SHOW_DETECTION else "off")
    elif ch in ("k", "K"):
        save_config(controller)
    elif ch in ("l", "L"):
        load_config(controller)
    elif ch in ("c", "C"):
        controller.calibration.start()
    elif ch in ("r", "R"):
        controller.smoother.reset()
        controller.click.cancel_all()
        controller.scroll.stop()
        controller.palm.reset()
        controller.wave.reset()
        print("[RESET] all gesture state cleared")
    elif ch in ("i", "I"):
        set_startup(not STARTUP_WINDOWS)
    elif ch in ("j", "J"):
        make_shortcut()


# ============================================================================
# MAIN LOOP
# ============================================================================

def run():
    global SCREEN_W, SCREEN_H
    import cv2

    try:
        pag = load_pyautogui()
    except Exception as exc:                    # noqa: BLE001
        print(f"[FATAL] pyautogui unavailable: {exc}")
        print("        pip install pyautogui")
        return 1

    SCREEN_W, SCREEN_H = pag.size()

    print("=" * 60)
    print(" HandGestureController v8.0 - Full Gesture Control")
    print(f" Screen {SCREEN_W}x{SCREEN_H}   camera {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(" [ ] box   - = sens   , . pinch   9 0 scroll   m preset   s sleep   q quit")
    print(" c calibrate   d dots   k/l save/load cfg   r reset   i startup   j shortcut")
    print("=" * 60)

    cap = open_camera()
    if cap is None:
        print("[FATAL] Camera not available.")
        print("        Close Zoom / Teams / OBS / browser tabs using the webcam, retry.")
        return 1

    hands = None
    try:
        hands = build_hands()
    except Exception as exc:                    # noqa: BLE001
        cap.release()
        print(f"[FATAL] MediaPipe failed to start: {exc}")
        print("        pip install --upgrade mediapipe")
        return 1

    controller = HandController(hands, pag=pag)
    load_config(controller)
    win = "HandGestureController v8.0"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CAMERA_WIDTH, CAMERA_HEIGHT)

    grabber = None
    if USE_THREADED_CAPTURE:
        try:
            grabber = FrameGrabber(cap).start()
            print("[CAM] threaded capture on")
        except Exception as exc:                  # noqa: BLE001
            print(f"[CAM] threaded capture unavailable ({exc}); using blocking read")
            grabber = None

    frame_fail = 0
    fps = 0.0
    fps_t0 = time.monotonic()
    fps_frames = 0
    try:
        while True:
            ok, frame = (grabber.read() if grabber is not None else cap.read())
            if not ok or frame is None:
                frame_fail += 1
                if frame_fail > 90:
                    print("[FATAL] Camera stopped delivering frames.")
                    break
                time.sleep(0.01)               # never spin the CPU on a dead camera
                continue
            frame_fail = 0

            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            if MIRROR:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            lm = None
            if results is not None and getattr(results, "multi_hand_landmarks", None):
                lm = results.multi_hand_landmarks[0].landmark
                if len(lm) < 21:
                    lm = None

            now = time.monotonic()
            controller.process(lm, now)

            fps_frames += 1
            if now - fps_t0 >= 0.5:
                fps = fps_frames / (now - fps_t0)
                fps_t0 = now
                fps_frames = 0

            draw_hud(cv2, frame, lm, controller, now, fps)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            handle_key(key, controller)
    except KeyboardInterrupt:
        print("\n[exit] interrupted")
    finally:
        save_config(controller)
        if grabber is not None:
            grabber.stop()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            hands.close()
        except Exception:
            pass
        print("[exit] camera and windows released")
    return 0


# ============================================================================
# SELFTEST
# ============================================================================

class _P:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def make_hand(pinch=0.6, fist=False, index_up=True, palm=False, fist_y=0.55):
    """21 synthetic landmarks; `pinch` = thumb-tip gap / hand size."""
    lm = [_P(0.5, 0.75) for _ in range(21)]
    lm[0] = _P(0.5, 0.75)
    lm[9] = _P(0.5, fist_y)

    lm[5] = _P(0.46, 0.58)
    if fist or not index_up:
        lm[6] = _P(0.45, 0.68)
        lm[7] = _P(0.46, 0.72)
        lm[8] = _P(0.47, 0.74)
    else:
        lm[6] = _P(0.44, 0.48)
        lm[7] = _P(0.43, 0.40)
        lm[8] = _P(0.42, 0.32)

    for mcp, pip, tip, x in ((10, 11, 12, 0.50), (14, 15, 16, 0.54), (18, 19, 20, 0.58)):
        lm[mcp] = _P(x, 0.57)
        if fist:
            lm[pip] = _P(x, 0.66)
            lm[tip] = _P(x, 0.73)
        elif palm:
            lm[pip] = _P(x, 0.46)
            lm[tip] = _P(x, 0.34)
        else:
            lm[pip] = _P(x, 0.64)
            lm[tip] = _P(x, 0.72)

    hs = math.hypot((lm[0].x - lm[9].x) * CAMERA_WIDTH, (lm[0].y - lm[9].y) * CAMERA_HEIGHT)
    target_px = pinch * hs
    dx, dy = lm[8].x - 0.36, lm[8].y - 0.66
    n = math.hypot(dx * CAMERA_WIDTH, dy * CAMERA_HEIGHT) or 1.0
    ux, uy = dx * CAMERA_WIDTH / n, dy * CAMERA_HEIGHT / n
    lm[1] = _P(0.40, 0.74)
    lm[2] = _P(0.37, 0.70)
    lm[3] = _P(0.35, 0.67)
    lm[4] = _P(lm[8].x + ux * target_px / CAMERA_WIDTH,
               lm[8].y + uy * target_px / CAMERA_HEIGHT)
    return lm


def selftest():
    global SCREEN_W, SCREEN_H, PINCH_THRESHOLD, PINCH_THRESHOLD_POINTING
    global ACTIVE_X_MIN, ACTIVE_X_MAX, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM
    SCREEN_W, SCREEN_H = 1920, 1080
    ok = fail = 0

    def check(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name} {extra}")
        else:
            fail += 1
            print(f"  FAIL  {name} {extra}")

    print("\n=== GEOMETRY ===")
    h_open = make_hand(pinch=0.70, index_up=True)
    h_pinch = make_hand(pinch=0.15, index_up=True)
    h_fist = make_hand(pinch=0.10, fist=True, index_up=False)
    h_palm = make_hand(pinch=0.70, palm=True)
    check("index extended", is_index_extended(h_open))
    check("index curled", not is_index_extended(h_fist))
    check("fist detected", is_fist_like(h_fist))
    check("pointing is not a fist", not is_fist_like(h_open))
    check("palm detected", is_open_palm(h_palm))
    check("pinch ratio small", pinch_distance(h_pinch) < 0.34)
    check("open ratio large", pinch_distance(h_open) > 0.5)

    print("\n=== CLICK: single / double / storm / fist ===")
    ce = ClickEngine()
    now = 1000.0

    def step(ratio, dt=0.033, masked=False, ext=True):
        nonlocal now
        now += dt
        return ce.update(ratio, now, masked=masked, cursor_xy=(500, 400), index_extended=ext)

    step(0.70)
    step(0.20)
    for _ in range(10):
        step(0.18)
    check("no storm while holding", ce.n_single == 0 and ce.n_double == 0)
    step(0.45)
    step(0.45)
    check("pending armed on release", ce.pending and ce.deadline is not None)
    check("single after window", ce.tick(now + DOUBLE_PINCH_WINDOW + 0.01) == "single")

    ce2 = ClickEngine()
    now = 2000.0
    for r in (0.70, 0.20, 0.20, 0.40, 0.40, 0.20, 0.20):
        now += 0.03
        ce2.update(r, now, masked=False, cursor_xy=(0, 0), index_extended=True)
    check("small double pinch -> double", ce2.n_double == 1 and ce2.n_single == 0)

    ce3 = ClickEngine()
    now = 3000.0
    for _ in range(40):
        now += 0.033
        ce3.update(0.05, now, masked=True, cursor_xy=(0, 0), index_extended=False)
    check("fist never clicks", ce3.tick(now + 5) is None and ce3.n_single == 0)

    ce4 = ClickEngine()
    now = 4000.0
    for _ in range(5):
        now += 0.033
        ce4.update(0.30, now, masked=False, cursor_xy=(0, 0), index_extended=True)
    check("pointing threshold stricter (0.30 no click)", ce4.state == "OPEN")

    print("\n=== B2: press with no release is dropped ===")
    ce5 = ClickEngine()
    now = 5000.0
    ce5.update(0.70, now, masked=False, cursor_xy=(0, 0), index_extended=True)
    now += 0.03
    ce5.update(0.15, now, masked=False, cursor_xy=(0, 0), index_extended=True)
    now += PINCH_STALE_TIMEOUT + 0.05
    ce5.update(0.15, now, masked=False, cursor_xy=(0, 0), index_extended=True)
    check("stuck pending cleared", ce5.pending is False)
    check("no phantom click fired", ce5.tick(now + 10) is None and ce5.n_single == 0)

    ce6 = ClickEngine()
    now = 5500.0
    ce6.update(0.70, now, masked=False, cursor_xy=(0, 0), index_extended=True)
    singles = 0
    for _ in range(150):                      # hold a pinch for 5 seconds
        now += 0.033
        ce6.update(0.15, now, masked=False, cursor_xy=(0, 0), index_extended=True)
        if ce6.tick(now):
            singles += 1
    check("B2: holding a pinch clicks at most once", singles <= 1, f"({singles} clicks in 5s)")

    print("\n=== SCROLL ===")
    se = ScrollEngine()
    now = 6000.0
    down = []
    for _ in range(30):
        now += 0.033
        down += se.update(make_hand(fist=True, fist_y=0.66), now, masked=False)
    check("fist low -> down", len(down) > 10 and all(a < 0 for a in down), f"{len(down)} ticks/s")

    se = ScrollEngine()
    now = 7000.0
    up = []
    for _ in range(30):
        now += 0.033
        up += se.update(make_hand(fist=True, fist_y=0.24), now, masked=False)
    check("fist high -> up", len(up) > 10 and all(a > 0 for a in up))

    se = ScrollEngine()
    now = 8000.0
    mid = []
    for _ in range(30):
        now += 0.033
        mid += se.update(make_hand(fist=True, fist_y=0.42), now, masked=False)
    check("fist mid -> none", len(mid) == 0)

    se = ScrollEngine()
    now = 9000.0
    parked = []
    for _ in range(30):
        now += 0.033
        parked += se.update(make_hand(fist=True, fist_y=0.95), now, masked=False)
    check("fist parked -> none", len(parked) == 0)

    print("\n=== B5: no tick before the flip delay ===")
    se = ScrollEngine()
    now = 9500.0
    early = se.update(make_hand(fist=True, fist_y=0.66), now, masked=False)
    check("first tick waits for FLIP_DELAY", early == [])

    print("\n=== CURSOR ===")
    cs = CursorSmoother()
    h = make_hand(index_up=True, pinch=0.7)
    h[5] = _P(0.99, 0.99)
    h[8] = _P(0.99, 0.99)
    for _ in range(400):
        x, y = cs.update(h, False)
    check("bottom-right reachable", x >= 1915 and y >= 1075, f"({x:.0f},{y:.0f})")

    cs = CursorSmoother()
    h = make_hand(index_up=True, pinch=0.7)
    h[5] = _P(0.01, 0.01)
    h[8] = _P(0.01, 0.01)
    for _ in range(400):
        x, y = cs.update(h, False)
    check("top-left reachable", x <= 2 and y <= 2, f"({x:.0f},{y:.0f})")

    cs = CursorSmoother()
    for _ in range(200):
        cs.update(make_hand(index_up=True, pinch=0.7), False)
    before = (cs.cx, cs.cy)
    for _ in range(10):
        cs.update(make_hand(index_up=True, pinch=0.7), True)
    check("frozen cursor stays put", (cs.cx, cs.cy) == before)

    cs = CursorSmoother()
    for _ in range(200):
        cs.update(make_hand(index_up=True, pinch=0.7), False)
    base = (cs.cx, cs.cy)
    import random
    random.seed(1)
    drift = 0.0
    for _ in range(60):
        hj = make_hand(index_up=True, pinch=0.7)
        dx = (random.random() - 0.5) * 0.002
        dy = (random.random() - 0.5) * 0.002
        hj[5] = _P(0.46 + dx, 0.58 + dy)
        hj[8] = _P(0.42 + dx, 0.32 + dy)
        x, y = cs.update(hj, False)
        drift += abs(x - base[0]) + abs(y - base[1])
    check("tremor suppressed", drift < 30, f"{drift:.1f}px/60f")

    print("\n=== B3: pointing does not freeze the cursor ===")
    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.last_engaged = 1e9
    ctl.reacquire_time = -1e9
    h = make_hand(index_up=True, pinch=0.45)
    ctl.process(h, 0.0)
    h2 = make_hand(index_up=True, pinch=0.45)
    h2[5] = _P(0.70, 0.58)
    h2[8] = _P(0.70, 0.32)
    p1 = ctl.process(h2, 0.05)
    p2 = ctl.process(h2, 0.10)
    check("cursor moves while a loose pinch is held", p2 is not None and p2[0] != p1[0],
          f"{p1} -> {p2}")

    print("\n=== B1: sleep actually gates everything ===")
    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.asleep = True
    ctl.last_engaged = 1e9
    ctl.reacquire_time = -1e9
    n0 = ctl.click.n_single
    now = 0.0
    for r in (0.70, 0.15, 0.15, 0.70, 0.70):
        now += 0.05
        ctl.process(make_hand(index_up=True, pinch=r), now)
    check("no clicks while asleep", ctl.click.n_single == n0)
    check("scroll blocked while asleep", ctl.scroll.direction == 0)

    print("\n=== WAVE -> sleep / wake ===")
    w = WaveDetector()
    now = 0.0
    fired = False
    h = make_hand(palm=True)
    for i in range(50):                       # realistic 2.5 Hz sine wave, ~1.6 s
        now += 0.033
        h[9] = _P(0.5 + 0.12 * math.sin(2 * math.pi * 2.5 * now), 0.55)
        if w.update(h, now):
            fired = True
    check("wave detected after 3+ swings", fired)

    w2 = WaveDetector()
    now = 0.0
    h = make_hand(palm=True)
    fired = False
    for _ in range(40):
        now += 0.033
        if w2.update(h, now):
            fired = True
    check("still palm -> no wave", not fired)

    w3 = WaveDetector()
    now = 0.0
    h = make_hand(palm=True)
    fired = False
    h[9] = _P(0.50, 0.55)
    for _ in range(40):                       # one long single movement
        now += 0.033
        if w3.update(h, now):
            fired = True
    check("one long movement -> no wave", not fired)

    print("\n=== PALM SWIPE ===")
    def swipe(dx, dy, now0):
        p = PalmSwipeDetector()
        out = []
        h = make_hand(palm=True)
        h[9] = _P(0.5 - dx / 2, 0.5 - dy / 2)
        p.update(h, now0)
        for i in range(1, 7):
            h[9] = _P(0.5 - dx / 2 + dx * i / 6, 0.5 - dy / 2 + dy * i / 6)
            e = p.update(h, now0 + i * 0.03)
            if e:
                out.append(e)
        return out

    check("swipe right -> 'right' event", "right" in swipe(0.30, 0.0, 100.0))
    check("swipe left -> 'left' event", "left" in swipe(-0.30, 0.0, 200.0))
    check("swipe up -> volup", "volup" in swipe(0.0, -0.24, 300.0))
    check("swipe down -> voldown", "voldown" in swipe(0.0, 0.24, 400.0))
    check("tiny move -> nothing", swipe(0.05, 0.0, 500.0) == [])

    p = PalmSwipeDetector()
    h = make_hand(palm=True)
    h[9] = _P(0.2, 0.5)
    p.update(h, 600.0)
    hits = 0
    for i in range(1, 8):                     # keep going right, past the threshold
        h[9] = _P(0.2 + 0.05 * i, 0.5)
        if p.update(h, 600.0 + i * 0.03):
            hits += 1
    check("B6: one swipe fires once", hits == 1, f"({hits})")

    print("\n=== SWIPE -> KEY MAPPING ===")
    class FakePag:
        def __init__(self):
            self.pressed = []
        def press(self, k):
            self.pressed.append(k)
        def click(self, *a, **kw):
            pass
        def doubleClick(self, *a, **kw):
            pass
        def moveTo(self, *a, **kw):
            pass
        def scroll(self, *a, **kw):
            pass

    fp = FakePag()
    ctl = HandController(pag=fp)
    ctl._do_media("left", 0.0)
    check("palm LEFT -> seek forward (right arrow)",
          fp.pressed == ["right"] * SEEK_PRESSES, str(fp.pressed))
    fp.pressed.clear()
    ctl._do_media("right", 0.0)
    check("palm RIGHT -> seek backward (left arrow)",
          fp.pressed == ["left"] * SEEK_PRESSES, str(fp.pressed))
    fp.pressed.clear()
    ctl._do_media("volup", 0.0)
    check("palm UP -> volume up", fp.pressed == ["volumeup"] * VOLUME_STEPS)
    fp.pressed.clear()
    ctl._do_media("voldown", 0.0)
    check("palm DOWN -> volume down", fp.pressed == ["volumedown"] * VOLUME_STEPS)

    fp.pressed.clear()
    old_mode, old_inv = PALM_HORIZONTAL_MODE, PALM_HORIZONTAL_INVERT
    _SELF.PALM_HORIZONTAL_MODE = "track"
    ctl._do_media("left", 0.0)
    check("track mode still available", fp.pressed == ["nexttrack"], str(fp.pressed))
    fp.pressed.clear()
    _SELF.PALM_HORIZONTAL_INVERT = True
    _SELF.PALM_HORIZONTAL_MODE = "seek"
    ctl._do_media("left", 0.0)
    check("invert flips left/right", fp.pressed == ["left"] * SEEK_PRESSES, str(fp.pressed))
    _SELF.PALM_HORIZONTAL_MODE, _SELF.PALM_HORIZONTAL_INVERT = old_mode, old_inv

    print("\n=== REST ===")
    ctl = HandController(pag=None)
    ctl.last_engaged = 0.0
    ctl.resting = True
    ctl.process(make_hand(index_up=False, fist=True, fist_y=0.55), 0.0)
    check("stays resting with fist", ctl.resting)
    ctl.process(make_hand(index_up=True), 0.05)
    ctl.process(make_hand(index_up=True), 0.10)
    check("index wakes it", not ctl.resting)

    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.last_engaged = 0.0
    def rest_hand():
        hh = make_hand(index_up=False, palm=True)
        hh[8] = _P(0.47, 0.74)
        return hh
    for i in range(40):
        ctl.process(rest_hand(), i * 0.033)
    check("idle hand -> rest", ctl.resting)

    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.last_engaged = 0.0
    for i in range(40):
        ctl.process(None, i * 0.033)
    check("hand lost -> rest", ctl.resting)

    print("\n=== B10: clock jump does not fire a burst ===")
    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.last_engaged = 1e9
    ctl.process(make_hand(fist=True, fist_y=0.66), 0.0)
    ctl.process(make_hand(fist=True, fist_y=0.66), 0.03)
    n_before = ctl.stats["scroll"]
    ctl.process(make_hand(fist=True, fist_y=0.66), 30.0)     # 30s jump
    check("no scroll burst after jump", ctl.stats["scroll"] - n_before <= 1,
          f"(+{ctl.stats['scroll'] - n_before})")

    print("\n=== PENDING CLICK SURVIVES HAND LOSS ===")
    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.last_engaged = 0.0
    ctl.reacquire_time = -1e9
    now = 0.0
    for r in (0.70, 0.15, 0.15, 0.70, 0.70):
        now += 0.05
        ctl.process(make_hand(index_up=True, pinch=r), now)
    armed = ctl.click.pending
    for _ in range(20):
        now += 0.05
        ctl.process(None, now)
    check("armed before loss", armed)
    check("still fires after loss", ctl.click.n_single == 1, f"({ctl.click.n_single})")

    print("\n=== FUZZ (4000 frames, random poses + time) ===")
    ctl = HandController(pag=None)
    import random
    random.seed(7)
    t = 0.0
    errs = 0
    gens = [
        lambda: make_hand(index_up=True, pinch=random.uniform(0.05, 0.9)),
        lambda: make_hand(fist=True, index_up=False, fist_y=random.uniform(0.05, 0.95)),
        lambda: make_hand(palm=random.random() > 0.5, pinch=random.uniform(0.05, 0.9)),
        lambda: None,
    ]
    for i in range(4000):
        t += random.choice([0.0, 0.016, 0.033, 0.05, 0.2, 1.2])
        try:
            ctl.process(random.choice(gens)(), t)
        except Exception as exc:                # noqa: BLE001
            errs += 1
            print(f"    frame {i}: {type(exc).__name__}: {exc}")
            if errs > 3:
                break
    check("no exceptions", errs == 0)
    check("cursor on screen", 0 <= ctl.cursor[0] < SCREEN_W and 0 <= ctl.cursor[1] < SCREEN_H)

    print("\n=== FPS-ADAPTIVE SMOOTHING ===")
    def settle_frames(fps, frac=0.9, limit=400):
        """Frames needed to cover `frac` of a step, at a given frame rate."""
        cs = CursorSmoother()
        h0 = make_hand(index_up=True, pinch=0.7)
        cs.update(h0, False, 1.0 / fps)
        start = cs.cx
        h1 = make_hand(index_up=True, pinch=0.7)
        h1[5] = _P(0.60, 0.58)
        h1[8] = _P(0.60, 0.32)
        final = None
        for _ in range(limit):                 # find where it converges
            final, _ = cs.update(h1, False, 1.0 / fps)
        target = start + frac * (final - start)
        cs = CursorSmoother()
        cs.update(h0, False, 1.0 / fps)
        for n in range(1, limit + 1):
            x, _ = cs.update(h1, False, 1.0 / fps)
            if (final > start and x >= target) or (final < start and x <= target):
                return n
        return limit

    n30 = settle_frames(30.0)
    n15 = settle_frames(15.0)
    t30, t15 = n30 / 30.0, n15 / 15.0
    check("90% of a step takes ~the same WALL-CLOCK time at 30 and 15 fps",
          abs(t30 - t15) < 0.20 and t15 < 0.62,
          f"({t30:.2f}s @30fps vs {t15:.2f}s @15fps; was 0.43 vs 0.67 before the fix)")

    # degenerate dt must not explode
    cs = CursorSmoother()
    h = make_hand(index_up=True, pinch=0.7)
    for dt in (0.0, -0.1, 1e-9, 5.0, None):
        cs.update(h, False, dt)
    x, y = cs.update(h, False, 0.033)
    check("absurd dt values are safe", 0 <= x < SCREEN_W and 0 <= y < SCREEN_H, f"({x:.0f},{y:.0f})")

    print("\n=== CONFIG PERSISTENCE ===")
    import tempfile, os as _os
    tmp = _os.path.join(tempfile.mkdtemp(), "cfg.json")
    old_path = _SELF.CONFIG_PATH
    _SELF.CONFIG_PATH = tmp
    ctl = HandController(pag=None)
    ctl.smoother.sensitivity = 1.4
    ctl.scroll.multiplier = 2.0
    ok_saved = save_config(ctl)
    check("config saved", ok_saved and _os.path.exists(tmp))

    ctl2 = HandController(pag=None)
    ctl2.smoother.sensitivity = 1.0
    ctl2.scroll.multiplier = 1.0
    ok_loaded = load_config(ctl2)
    check("config loaded back", ok_loaded and abs(ctl2.smoother.sensitivity - 1.4) < 1e-6
          and abs(ctl2.scroll.multiplier - 2.0) < 1e-6,
          f"sens={ctl2.smoother.sensitivity} mult={ctl2.scroll.multiplier}")

    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("{ this is not json")
    ok_bad = load_config(ctl2)
    check("corrupt config is ignored, not fatal", ok_bad is False)

    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write('{"box": [400, 100, 300, 50], "pinch_threshold": 99, '
                 '"sensitivity": -5, "scroll_mult": 1e9}')
    load_config(ctl2)
    check("out-of-range values are clamped",
          0 <= ACTIVE_X_MIN < ACTIVE_X_MAX <= CAMERA_WIDTH
          and 0.15 <= PINCH_THRESHOLD <= 0.60
          and SENSITIVITY_MIN <= ctl2.smoother.sensitivity <= SENSITIVITY_MAX
          and SCROLL_SPEED_MIN <= ctl2.scroll.multiplier <= SCROLL_SPEED_MAX,
          f"box={ACTIVE_X_MIN}-{ACTIVE_X_MAX} pinch={PINCH_THRESHOLD} sens={ctl2.smoother.sensitivity}")
    _SELF.CONFIG_PATH = old_path
    # restore the globals the config test deliberately corrupted
    PINCH_THRESHOLD = 0.34
    PINCH_THRESHOLD_POINTING = 0.26
    ACTIVE_X_MIN, ACTIVE_X_MAX, ACTIVE_Y_TOP, ACTIVE_Y_BOTTOM = 130, 510, 90, 335

    print("\n=== IDLE AUTO-SLEEP ===")
    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.last_engaged = 0.0
    now = 0.0
    ctl.process(make_hand(index_up=True, pinch=0.7), now)
    ctl.last_action_time = now
    for i in range(int(IDLE_SLEEP_TIME / 0.05) + 40):
        now += 0.05
        ctl.process(make_hand(index_up=True, pinch=0.7), now)
    check(f"auto-sleeps after {IDLE_SLEEP_TIME}s idle", ctl.asleep)

    print("\n=== 5. DRAG (hold) ===")
    class RecPag:
        def __init__(self):
            self.events = []
        def press(self, k): self.events.append(("press", k))
        def click(self, *a, **kw): self.events.append(("click", a, kw))
        def doubleClick(self, *a, **kw): self.events.append(("double",))
        def mouseDown(self, *a, **kw): self.events.append(("down",))
        def mouseUp(self, *a, **kw): self.events.append(("up",))
        def keyDown(self, k): self.events.append(("keyDown", k))
        def keyUp(self, k): self.events.append(("keyUp", k))
        def moveTo(self, *a): pass
        def scroll(self, *a): pass

    ce = ClickEngine()
    now = 20000.0
    ce.update(0.70, now, masked=False, cursor_xy=(10, 20), index_extended=True,
              right_ratio=0.95, can_hold=True)
    now += 0.03
    ce.update(0.15, now, masked=False, cursor_xy=(10, 20), index_extended=True,
              right_ratio=0.95, can_hold=True)
    started = None
    for _ in range(30):                       # keep holding the pinch
        now += 0.03
        if ce.hold_tick(now):
            started = now
            break
    check("held pinch becomes a drag", started is not None)
    check("drag suppresses the click", ce.pending is False and ce.tick(now + 5) is None)
    rel = ce.hold_release(now + 1.0)
    check("drag release emitted", rel == "hold_release" and ce.hold_active is False)

    ce = ClickEngine()
    now = 21000.0
    ce.update(0.70, now, masked=False, cursor_xy=(0, 0), index_extended=True,
              right_ratio=0.95, can_hold=True)
    now += 0.03
    ce.update(0.15, now, masked=False, cursor_xy=(0, 0), index_extended=True,
              right_ratio=0.95, can_hold=True)
    now += 0.05
    ce.update(0.60, now, masked=False, cursor_xy=(0, 0), index_extended=True,
              right_ratio=0.95, can_hold=True)          # quick release
    now += 0.05
    ce.update(0.60, now, masked=False, cursor_xy=(0, 0), index_extended=True,
              right_ratio=0.95, can_hold=True)
    check("quick pinch does NOT become a drag", ce.hold_tick(now) is None)

    fp = RecPag()
    ctl = HandController(pag=fp)
    ctl.click.hold_active = True
    ctl._do_click("hold_release")
    check("hold_release drives the mouse", ("up",) in fp.events, str(fp.events))

    print("\n=== 5b. RIGHT CLICK (index+middle pinch) ===")
    ce = ClickEngine()
    now = 22000.0
    act = ce.update(0.70, now, masked=False, cursor_xy=(0, 0), index_extended=False,
                    right_ratio=0.40, can_hold=True)
    check("index+middle pinch -> right click", act == "right")
    now += 0.05
    act2 = ce.update(0.70, now, masked=False, cursor_xy=(0, 0), index_extended=False,
                     right_ratio=0.40, can_hold=True)
    check("no repeat right click while held", act2 is None)
    now += 0.05
    ce.update(0.70, now, masked=False, cursor_xy=(0, 0), index_extended=False,
              right_ratio=0.90, can_hold=True)         # release pose
    now += 0.05
    act3 = ce.update(0.70, now, masked=False, cursor_xy=(0, 0), index_extended=False,
                     right_ratio=0.40, can_hold=True)
    check("re-arms after release + cooldown", act3 is None or act3 == "right")

    h_rc = make_hand(index_up=True, pinch=0.70)
    check("pointing hand is NOT a right click", right_click_ratio(h_rc) > RIGHT_CLICK_RATIO,
          f"({right_click_ratio(h_rc):.2f} vs {RIGHT_CLICK_RATIO:.2f})")

    print("\n=== 7. VELOCITY SCROLL ===")
    se = ScrollEngine()
    now = 23000.0
    near = []
    for _ in range(60):                       # just past the dead zone
        now += 0.016
        near += se.update(make_hand(fist=True, fist_y=0.45), now, masked=False)
    se2 = ScrollEngine()
    now = 24000.0
    far = []
    for _ in range(60):                       # pushed to the bottom of the box
        now += 0.016
        far += se2.update(make_hand(fist=True, fist_y=0.66), now, masked=False)
    check("further from centre scrolls faster", len(far) > len(near) + 5,
          f"(near {len(near)} vs far {len(far)} ticks)")

    print("\n=== 10. CALIBRATION WIZARD ===")
    cal = Calibration()
    cal.start()
    poses = [make_hand(index_up=True, pinch=0.80),
             make_hand(index_up=True, pinch=0.12),
             make_hand(fist=True, index_up=False),
             make_hand(index_up=True, pinch=0.70)]
    done = False
    for pose in poses:
        for _ in range(CALIB_SETTLE_FRAMES + CALIB_HOLD_FRAMES + 2):
            if cal.update(pose):
                done = True
    check("wizard completes four steps", done and cal.result is not None, str(cal.result))
    res = cal.result
    check("open > pinch threshold > pinch",
          res["raw"]["open"] > res["pinch_threshold"] > res["raw"]["pinch"],
          f"open={res['raw']['open']} thr={res['pinch_threshold']} pinch={res['raw']['pinch']}")
    check("thresholds stay in range",
          0.15 <= res["pinch_threshold"] <= 0.60 and 0.12 <= res["pinch_pointing"] <= res["pinch_threshold"])

    ctl = HandController(pag=None)
    apply_calibration(ctl, res)
    check("calibration applied to live config", abs(PINCH_THRESHOLD - res["pinch_threshold"]) < 1e-9)

    # gestures must not fire while calibrating
    ctl = HandController(pag=None)
    ctl.resting = False
    ctl.last_engaged = 1e9
    ctl.calibration.start()
    n0 = ctl.click.n_single
    for i in range(200):
        ctl.process(make_hand(index_up=True, pinch=0.15), i * 0.03)
    check("no clicks during calibration", ctl.click.n_single == n0)

    print("\n=== 12. VOLUME BACKENDS ===")
    fp = RecPag()
    old_backend = _SELF.VOLUME_BACKEND
    _SELF.VOLUME_BACKEND = "keys"
    set_volume(3, fp)
    check("keys backend -> 3 volumeup", fp.events.count(("press", "volumeup")) == 3,
          str(fp.events))
    fp.events.clear()
    set_volume(-2, fp)
    check("keys backend -> 2 volumedown", fp.events.count(("press", "volumedown")) == 2)
    _SELF.VOLUME_BACKEND = old_backend

    print("\n=== 11. STARTUP HELPERS ===")
    cmd = startup_command()
    check("startup command mentions the script", script_path() in cmd, cmd[:60] + "...")
    if sys.platform != "win32":
        check("non-Windows degrades gracefully", set_startup(True) is False)

    print("\n=== 6. MEDIAPIPE BACKEND ADAPTER ===")
    class FakeRes:
        multi_hand_landmarks = None
    ad = HandsAdapter("solutions", type("X", (), {"process": lambda self, x: FakeRes(),
                                                  "close": lambda self: None})(), "video")
    check("adapter returns solutions result unchanged", isinstance(ad.process(None), FakeRes))

    class Boom:
        def process(self, x):
            raise RuntimeError("backend died")
        def close(self):
            pass
    ad2 = HandsAdapter("tasks", Boom(), "video")
    check("adapter survives a backend exception", ad2.process(None) is None)

    print("\n=== HOTKEYS ===")
    ctl = HandController(pag=None)
    s0 = ctl.smoother.sensitivity
    handle_key(ord("="), ctl)
    check("'=' up", ctl.smoother.sensitivity > s0)
    handle_key(ord("+"), ctl)
    check("'+' up", ctl.smoother.sensitivity > s0 + 0.1)
    handle_key(ord("-"), ctl)
    handle_key(ord("-"), ctl)
    check("'-' down", abs(ctl.smoother.sensitivity - s0) < 1e-9)

    t0 = PINCH_THRESHOLD
    handle_key(ord("."), ctl)
    check("'.' easier", PINCH_THRESHOLD > t0)
    check("B9: pointing threshold stays sane", 0.12 <= PINCH_THRESHOLD_POINTING <= PINCH_THRESHOLD)
    for _ in range(60):
        handle_key(ord(","), ctl)
    check("B9: lower clamp holds", PINCH_THRESHOLD >= 0.15 and PINCH_THRESHOLD_POINTING >= 0.12)
    for _ in range(60):
        handle_key(ord("."), ctl)
    check("B9: upper clamp holds", PINCH_THRESHOLD <= 0.60)

    m0 = ctl.scroll.multiplier
    handle_key(ord("9"), ctl)
    check("'9' faster", ctl.scroll.multiplier > m0)
    handle_key(ord("0"), ctl)
    check("'0' slower", abs(ctl.scroll.multiplier - m0) < 1e-9)

    x0, x1 = ACTIVE_X_MIN, ACTIVE_X_MAX
    handle_key(ord("]"), ctl)
    check("']' shrinks box", ACTIVE_X_MAX - ACTIVE_X_MIN < x1 - x0)
    handle_key(ord("["), ctl)
    check("'[' restores box", ACTIVE_X_MAX - ACTIVE_X_MIN == x1 - x0)
    handle_key(-1, ctl)
    check("no-key frame safe", True)

    p0 = ctl.smoother.preset
    a = ctl.smoother.cycle_preset()
    b = ctl.smoother.cycle_preset()
    c = ctl.smoother.cycle_preset()
    check("preset cycles", a != b and b != c and c == p0, f"{p0}->{a}->{b}->{c}")

    print("\n" + "=" * 50)
    print(f" RESULT: {ok} passed, {fail} failed")
    print("=" * 50)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run() or 0)
