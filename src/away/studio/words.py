"""Every word the Gesture Studio shows, in plain language (no ML jargon), plus friendly error explanations.

Kept apart from the widgets so a test can check the wording: no "confidence", "model", "takes", "blocked"...
"""

from __future__ import annotations

from away.studio.pack import NOTHING

TAB_TITLE = "Gesture Studio"

STATE_WORDS = {
    "idle": "Waiting",
    "arming": "Getting ready...",
    "recognized": "Seen (not used here)",
    "firing": "Doing it!",
    "cooldown": "Resting",
}

LABELS = {
    "camera_on": "Turn camera on",
    "camera_off": "Turn camera off",
    "details": "Show details",
    "practice_mode": "Practice mode (nothing really happens)",
    "how_sure": "How sure it is",
    "not_sure": "Not sure",
    "nothing": "No gesture",
    "teach": "Teach a new gesture",
    "practise_more": "Practise more",
    "learn": "Learn my gestures now",
    "gesture_settings": "Change what it does...",
    "remove": "Remove",
    "share": "Share my gestures...",
    "add_file": "Add gestures from a file...",
    "delete_recordings": "Delete all my recordings",
    "improve_builtin": "Improve the built-in gestures...",
    "works_here": "Works here",
    "ignore_here": "Ignore here",
    "draw_box": "Draw a box",
    "add_line": "Only above a line",
    "clear_zones": "Works everywhere",
    "grid": "Squares",
    "save_zones": "Save the areas",
    "zones_for": "Areas for",
    "my_gestures": "My gestures",
    "builtin_gestures": "Built-in gestures",
    "watching": "Watch",
    "thats_wrong": "That was wrong",
    "last_action": "Last thing it did",
    "examples": "Practice examples",
    "looks_like": "Looks most like",
    "timeline": "The last 5 seconds",
    "important_points": "Points it pays attention to",
    "mixups": "What it mixed up when checking itself",
    "where": "Where gestures work",
    "sees": "What it sees",
    "your_gestures": "Your gestures",
}

TOOLTIPS = {
    "camera_on": "Shows your camera picture with your hand drawn on top. Project Away pauses while this is on.",
    "details": "Shows extra charts: the last 5 seconds, the points of your hand it looks at, and what it mixes up.",
    "practice_mode": "When ticked, gestures are shown but no keys are pressed and nothing opens. "
    "Untick to try for real.",
    "how_sure": "How sure the app is that you are making this gesture right now. The red mark is how sure it must be.",
    "teach": "Three easy steps: name your gesture, show it a few times, try it.",
    "practise_more": "Record a few more practice examples of the selected gesture. More rounds make it more reliable.",
    "learn": "Teaches the app all your gestures again from your practice examples. It only needs a few seconds.",
    "gesture_settings": "Pick what happens when you make the selected gesture: a key, a click, typing text, "
    "opening an app or website, or a voice action.",
    "remove": "Removes the selected gesture and its practice examples.",
    "share": "Saves your gestures to a file you can give to someone else. Your recordings are only "
    "included if you tick the box.",
    "add_file": "Adds gestures from a file someone shared with you.",
    "delete_recordings": "Deletes every practice recording from this computer. Your taught gestures keep working "
    "until you teach again.",
    "improve_builtin": "Record your own examples of a built-in gesture (like swipe left) so it works better for you.",
    "grid": "How many squares the picture is split into. Click a square to switch it between 'Works here' and "
    "'Ignore here'. Drag the lines between squares to resize them.",
    "works_here": "Then click squares where gestures should work.",
    "ignore_here": "Then click squares where gestures should be ignored (for example where your coffee cup is).",
    "draw_box": "Then drag on the picture to draw a box. Gestures are ignored inside it.",
    "add_line": "Gestures only count when your hand is above this line. Click the picture to move the line.",
    "clear_zones": "Makes gestures work everywhere in the picture again.",
    "save_zones": "Remembers these areas for this camera and the chosen app.",
    "zones_for": "The areas can be different for each app. 'default' means every app without its own areas.",
    "thats_wrong": "Did something happen that you did not want? Press this and the last 2 seconds become an "
    "example of 'no gesture', so it learns not to do that.",
    "watching": "'My gestures' are the ones you taught. 'Built-in gestures' are the ones that come with the app.",
    "looks_like": "Which of your practice examples the current hand looks most like.",
    "timeline": "Each line is one gesture. Higher means the app is more sure. Marks at the bottom show what it did.",
    "important_points": "Bigger circles are the points of your hand that matter most for what it sees now.",
    "mixups": "Each row is what you did, each column is what the app thought. The diagonal is right.",
    "threshold": "Higher means it must be more sure before doing anything (fewer mistakes, but it may miss some).",
    "hold": "How long you must hold the gesture before it counts. Helps avoid accidents.",
    "rest": "After it does something, it waits this long before it can do it again.",
    "enabled": "Untick to switch this gesture off without deleting it.",
    "then": "Optional extra steps after the main action, one per line, like 'key:enter'. "
    "Use 'wait:0.5' to pause half a second.",
    "action": "What happens when you make this gesture.",
    "action_detail": "For a key: a key or combination like ctrl+c. For text: the words to type. For an app: its "
    "name or path. For a website: its address.",
}

KIND_WORDS = {"static": "A pose you hold still", "motion": "A movement"}
HANDS_WORDS = {1: "One hand", 2: "Two hands"}
# (what the user picks, action kind, fixed argument choices or None for free text)
ACTION_CHOICES: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("Do nothing", "none", ()),
    ("Press a key", "key", None),
    ("Click the mouse", "mouse", ("click", "double_click", "right_click", "middle_click", "scroll_up", "scroll_down")),
    ("Type some text", "type", None),
    ("Open an app", "launch", None),
    ("Open a website", "url", None),
    ("Start or stop dictation", "dictation", ("toggle", "start", "stop")),
    ("Voice: switch mode", "voice", ("toggle_mode", "command_mode", "dictation_mode")),
    ("Microphone on or off", "mic", ("toggle", "on", "off")),
    ("Pause gestures", "pause", ("toggle", "on", "off")),
    ("Run a command (advanced)", "shell", None),
)

WIZARD = {
    "title": "Teach a new gesture",
    "step1": "Step 1 of 3: Name it",
    "step1_help": "Give your gesture a short name, like 'thumbs up' or 'wave'.",
    "step2": "Step 2 of 3: Show me",
    "step2_help": "Make your gesture each time it says 'Now!'. Between rounds, just relax and move your hand "
    "naturally. Move a little between rounds (closer, further, a bit to the side) so it learns it well.",
    "step3": "Step 3 of 3: Try it",
    "step3_help": "Make your gesture and watch it light up. Then choose what it should do.",
    "start": "Start",
    "next": "Next",
    "back": "Back",
    "finish": "Finish",
    "cancel": "Cancel",
    "bad_one": "That one was bad, redo it",
    "relax": "Relax: let your hand move naturally",
    "get_ready": "Get ready...",
    "go": "Now!",
    "learning": "Learning your gesture...",
}


def name_for(cls: str) -> str:
    """'thumbs_up' -> 'Thumbs up'; the background class reads 'No gesture'."""
    if cls == NOTHING:
        return LABELS["nothing"]
    return cls.replace("_", " ").strip().capitalize()


def state_text(state: str) -> str:
    return STATE_WORDS.get(state, state)


def sure_text(p: float) -> str:
    return f"{round(100 * max(0.0, min(1.0, p)))}%"


def score_text(score: float | None, n_checked: int) -> str:
    if score is None or n_checked == 0:
        return "Practise in 3 or more rounds and it will tell you how well it knows your gestures."
    right = round(score * n_checked)
    return f"When checking itself on practice rounds it had not seen, it got {right} of {n_checked} right."


def summary_split(split: dict[str, object]) -> str:
    rounds = split.get("rounds_per_gesture") or {}
    if not isinstance(rounds, dict) or not rounds:
        return ""
    parts = [f"{name_for(str(k))}: {v} round{'s' if v != 1 else ''}" for k, v in rounds.items()]
    return "Practice rounds: " + ", ".join(parts)


def action_text(action: str) -> str:
    """'key:ctrl+c' -> 'press ctrl+c' (plain words for the 'Last thing it did' line)."""
    kind, _, arg = action.partition(":")
    words = {
        "none": "nothing (no action chosen yet)",
        "key": f"press {arg}",
        "mouse": arg.replace("_", " "),
        "type": f"type “{arg}”",
        "launch": f"open {arg}",
        "shell": f"run {arg}",
        "dictation": f"dictation {arg}",
        "mic": f"microphone {arg}",
        "voice": f"voice {arg.replace('_', ' ')}",
        "pause": f"pause {arg}",
        "hands": f"use {arg} hand",
        "cursor": f"cursor {arg.replace('_', ' ')}",
    }
    return words.get(kind, action)


# ---------------------------------------------------------------------- friendly errors with a fix button
FIXES = {
    "open_cameras": "Open camera settings",
    "retry": "Try again",
    "camera_on": "Turn camera on",
    "record_relax": "Record a relax round",
    "practise_more": "Practise more",
    "teach": "Teach a new gesture",
    "install_training": "Show me how",
    "allow_shell": "Show me how",
    "setup": "Show me how",
}

CAMERA_DID_NOT_START = "The camera did not start. Check it is plugged in and chosen in camera settings."
_EXPLAIN = (
    ("busy", "", "Another app is using the camera. Close it, then try again.", "retry"),
    ("in use", "", "Another app is using the camera. Close it, then try again.", "retry"),
    ("waiting for project away", "", "Project Away is letting go of the camera. This needs a few seconds.", ""),
    ("did not open", "", CAMERA_DID_NOT_START, "open_cameras"),
    ("could not open", "", CAMERA_DID_NOT_START, "open_cameras"),
    ("no frames", "", "The camera is not sending a picture. Check camera settings.", "open_cameras"),
    ("hand model missing", "", "Some hand-tracking files are missing.", "setup"),
    ("hand tracking could not start", "", "Hand tracking could not start.", "setup"),
    ("relax round", "", "The app also needs to see your hand doing nothing special.", "record_relax"),
    ("not in view", "", "Your hand was not in the picture. Move it into view and try again.", "retry"),
    ("round(s)", "practised in", "It needs a few more practice rounds to check itself.", "practise_more"),
    ("teach a gesture first", "", "You have not taught any gestures yet.", "teach"),
    ("no gestures taught", "", "You have not taught any gestures yet.", "teach"),
    ("learning add-on", "", "Improving the built-in gestures needs an extra add-on.", "install_training"),
    ("shell actions are disabled", "", "Running commands is switched off for safety.", "allow_shell"),
    ("built-in gesture model is missing", "", "The built-in gestures file is missing.", "setup"),
)


def explain(text: str) -> tuple[str, str, str] | None:
    """A worker/trainer message -> ``(friendly message, fix id, fix button text)``; None when it is not a
    problem the user can fix. The fix id may be "" (no button: it resolves itself)."""
    low = text.lower()
    for a, b, msg, fix in _EXPLAIN:
        if a in low and (not b or b in low):
            return msg, fix, FIXES.get(fix, "")
    return None


HELP = {
    "install_training": "To improve the built-in gestures, install the learning add-on once. Open a terminal and "
    'run:\n\n    pip install "project-away[train]"\n\nThen come back and press the button again.',
    "setup": "Open a terminal and run:\n\n    away setup\n\nIt downloads the hand-tracking files again.",
    "allow_shell": "Running commands is off for safety. To allow it, open your settings file and set\n\n"
    "    bindings:\n      allow_shell: true\n\nOnly do this if you trust every gesture pack you add.",
}
