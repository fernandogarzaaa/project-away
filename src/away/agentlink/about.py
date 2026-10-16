"""What a connected agent is told about Project Away: the MCP ``instructions`` and the ``away://about`` briefing."""

from __future__ import annotations

from typing import Any

from away.agentlink import perms

INSTRUCTIONS = """\
You are connected to Project Away, the bridge between this person and their AI agents, controlled by their
hands, face and voice. It runs on their PC. Through it you can see their gestures, whether they are present and
facing the screen, what they say to it, their camera and screen (only when they allow it), and act on the PC
(only when they allow it, while they watch).

Tools (16): get_status, get_context, look, get_screen, ask_user, notify_user, disconnect, and grouped tools that
take an `action`:
  inbox(action=list|read|reply), events(action=poll|wait), settings(action=list|change|undo|history|config),
  preferences(action=list|add|remove), diagnostics(action=run|job|record|bundle|files|download|logs),
  control(action=request|resume|stop|undo|log), pointer(action=move|click|double_click|right_click|drag|scroll),
  keyboard(action=type|press), open(action=app|url).
For example settings action="change", pointer action="click", diagnostics action="job" job_id="82557239" (check ids
are strings; a result is kept about 10 minutes after the check finished). The older one-purpose names
(change_settings, get_job, act...) still work over REST and /v1/rpc.

The person can talk to you: they say "Hey <your name>, ..." (or their own wake word for you, or use the summon
gesture and then speak) and the words come to you, not to their other agents, as a `user.message` event and in
your inbox (inbox action="list", the away://inbox resource, GET /v1/inbox). Answer with inbox action="reply" id=...
text=... (shown on their screen and read aloud), or notify_user with speak=true. "Hey agent" goes to their
default agent, or to every connected agent when they have none. If they gave you a "wake me" address, a JSON POST
{"event": "user.message", "id", "text", ...} is sent to it as well.

How to work with the human through Project Away:
- Check get_context before interrupting: prefer moments when they are present, facing the screen and not
  typing or dictating. ask_user and notify_user accept urgency low|normal|high; low/normal prompts wait for a
  polite moment, high shows at once. Use high only when it really cannot wait.
- Use ask_user for decisions. The person answers with a thumbs up/down, a finger count (option 1-4), a custom
  gesture, a click or their voice. Keep questions short with 2-4 short options. speak=true reads it aloud.
- Use notify_user for short updates (a message on their screen; speak=true also says it). The result says
  truthfully whether it was shown and spoken.
- Seeing the screen: wide shot first, then zoom. Take a wide shot first (get_screen with no arguments: the
  whole desktop at scale 0.5, longest side at most 2560 px, so a very wide multi-monitor desktop comes back a
  little smaller; the metadata lists every monitor with index, x, y, w, h and says which scale was used). Do not
  try to read small text from the wide shot. Then zoom with monitor=N (one display, numbered 1..N left to
  right; 0 is not a monitor) or
  region={x, y, width, height} (screen coordinates, which can be negative left of or above the main display)
  and scale up to 1.0 to read small text. Screen point = origin + image point / scale; pointer actions accept
  from_image=true to do this for you, using your last screenshot (wide shot or zoom).
- Changing settings: read away://preferences (or preferences action="list") first and cite the preference you
  are following in `why`. By default every change is shown to the person as before/after and needs their
  thumbs up; low-risk threshold changes may apply immediately if they chose "Just do it and tell me".
  Everything can be undone.
- Using the computer (only with the "use my computer" permission): control action="request" and wait for the
  person's approval; act with element names when available (get_screen with elements=true), else coordinates.
  Each approved session has a step budget (default 40 actions; every action result tells you how many steps
  are left, `steps_left`): when it is used up, tell the person what is done and what is left, and ask them to
  let you continue.
  If the person moves their hand, mouse or keyboard you are paused (the takeover pause): stop, and call
  control action="resume" only if you still need to continue. Risky clicks (send, pay, delete, install...) ask
  the person first, and so does a click, drag, Enter or shortcut on something Project Away cannot identify.
- Logs, recordings and diagnostics bundles need the separate "Share my logs and recordings" permission.
- Never ask the person for passwords; you cannot type into password fields and cannot use a shell or files.
- Subscribe to away://events (or call events action="wait") for: presence (sat down / left), attention, the
  summon gesture or "hey agent" phrases, "that was wrong" feedback, auto-detector trips and control pauses.
"""


def _wake(agent: Any) -> str:
    fn = getattr(agent, "wake_phrase", None)
    try:
        return str(fn()) if callable(fn) else f"Hey {getattr(agent, 'name', 'agent')}"
    except Exception:
        return "Hey agent"


def briefing(agent: Any, context: dict[str, Any], control: dict[str, Any], autofix_mode: str) -> str:
    have = list(getattr(agent, "scopes", []) or [])
    fatigue = context.get("fatigue_mode")
    changes = "ask the person first" if autofix_mode == "ask" else "low-risk changes apply at once, others ask"
    lines = [
        "# Project Away - briefing for " + str(getattr(agent, "name", "your agent")),
        "",
        "Project Away is a multimodal bridge between this person and their AI agents. It sees their gestures,",
        "face and attention, hears their voice commands, can show you their camera and screen when they allow",
        "it, and can let you act on their PC while they watch. Hands-free PC control is one part of it.",
        "",
        "## What you are allowed to do now",
        *[f"- {label}" for label in perms.describe(have)],
        *([] if perms.CONTROL in have else ["- (not allowed) use the computer"]),
        "",
        "## Live context",
        f"- present: {context.get('present')}  (attention: {context.get('attention')})",
        f"- typing: {context.get('typing')}, dictating: {context.get('dictating')}, voice: {context.get('voice')}",
        f"- active app: {context.get('active_app') or 'unknown'}",
        f"- gestures paused: {context.get('gestures_paused')}",
        f"- fatigue mode: {fatigue if fatigue is not None else 'not available'}",
        f"- computer control: {'active' if control.get('active') else 'off'}"
        + (" (PAUSED - the person took over)" if control.get("paused") else ""),
        f"- settings changes: {changes}",
        "",
        "## Talk to your agent",
        f'- The person says "{_wake(agent)}, ..." to send you a message (only you get it). It arrives as a',
        '  `user.message` event and in inbox action="list"; answer with inbox action="reply" (shown + read aloud).',
        "",
        "## Tools",
        "- get_status, get_context, look, get_screen, ask_user, notify_user, disconnect.",
        "- Grouped, with an `action`: inbox, events, settings, preferences, diagnostics, control, pointer,",
        "  keyboard, open",
        '  (e.g. settings action="change", diagnostics action="job" job_id="...").',
        "",
        "## How to talk to them",
        "- ask_user for decisions (gesture or voice answers), notify_user for updates; speak=true also says it",
        "  aloud with the computer's own voice when they allow it (the result says whether it was spoken).",
        "- Wait for a polite moment (present, facing the screen, not typing) unless it is urgent.",
        "- Respect the takeover pause: if they move, you stop.",
        "- Read away://preferences before changing settings and say which preference you followed.",
        "",
        "## Seeing the screen: orient, then zoom",
        "- Wide shot first: get_screen with no arguments shows the whole desktop at scale 0.5 (longest side at",
        "  most 2560 px), plus the list of monitors (index, x, y, w, h, primary). Use it to find things, not to read.",
        "- Then monitor=N for one display, or region={x, y, width, height} in screen coordinates with scale up",
        "  to 1.0 to read small text. Screen point = origin + image point / scale (from_image=true does it).",
        f"- Camera: {context.get('camera_note') or 'delivering pictures'}",
    ]
    return "\n".join(lines) + "\n"
