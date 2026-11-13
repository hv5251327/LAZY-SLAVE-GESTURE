# Lazy Slave 2 - Hand Gesture Controller v8.0

Everything, one hand only.

---

## ✋ Gestures

| Gesture | Action | Description |
| :--- | :--- | :--- |
| ☝️ **Show Index** | **Wake from REST** | Requires 2 consecutive index-up frames; nothing works while resting |
| ☝️ **Move Index** | **Move Cursor** | FPS-adaptive velocity smoothing with ergonomic active tracking box |
| 🤏 **Small Pinch x1** | **Single Click** | Closing edge (< 34% gap). Single click is deferred until release window expires |
| 🤏🤏 **Small Pinch x2** | **Double Click** | Second pinch within 0.45s fires double click immediately |
| 🤏⏱️ **Pinch + Hold (0.35s)** | **Drag & Drop** | Hold pinch to grab/drag (mouseDown or Shift); open pinch to drop |
| ✌️ **Index + Middle Pinch** | **Right Click** | Bringing index and middle fingers together triggers right-click |
| ✊ **Fist High / Low** | **Velocity Scroll** | Scroll up/down; tick rate scales with distance from center |
| 🖐️ **Open Palm Swipe LEFT** | **Seek Forward ⏩** | Press right arrow 2x (YouTube-style seek +10s forward) |
| 🖐️ **Open Palm Swipe RIGHT** | **Seek Backward ⏪** | Press left arrow 2x (YouTube-style seek -10s backward) |
| 🖐️ **Open Palm Move UP** | **Volume Up ▲** | Native volume scalar or media keys |
| 🖐️ **Open Palm Move DOWN** | **Volume Down ▼** | Native volume scalar or media keys |
| 🖐️ **Wave Palm (3 swings)** | **Sleep / Wake Toggle** | Fast 3-reversal wave toggles deep sleep mode |

---

## ⌨️ Keys (Preview Window Focused)

| Key | Action | Description |
| :---: | :--- | :--- |
| `[` / `]` | **Box Reach** | Make active box larger or smaller |
| `-` / `=` | **Sensitivity** | Decrease / increase cursor movement sensitivity |
| `,` / `.` | **Pinch Tuning** | Make pinch stricter (fewer false clicks) / easier (clicks sooner) |
| `9` / `0` | **Scroll Speed** | Faster / slower continuous fist scroll speed multiplier |
| `m` | **Smoothing Preset** | Cycle through presets: `BALANCED` ➔ `FAST` ➔ `PRECISION` |
| `s` | **Manual Sleep** | Toggle sleep/awake state manually |
| `d` | **Landmark Dots** | Toggle 21 landmark point display |
| `c` | **Calibrate** | Interactive 4-step hand calibration wizard |
| `k` / `l` | **Config Save / Load** | Save or load persistent config (`~/.hand_controller_v7.json`) |
| `r` | **Reset** | Clear all gesture and filter states |
| `i` | **Startup on Login** | Toggle Windows HKCU Run startup entry |
| `j` | **Desktop Shortcut** | Create desktop shortcut `.lnk` |
| `q` / `Esc` | **Quit** | Safely release camera and close windows |

---

## 🧪 Headless Automated Testing

```bash
python hand_cursor_controller.py --selftest
```
Output: `92 passed, 0 failed`
