"""What a connected agent is told about Project Away: the MCP ``instructions`` and the ``away://about`` briefing."""

from __future__ import annotations

from typing import Any

from away.agentlink import perms

INSTRUCTIONS = """\
You are connected to Project Away, the bridge between this person and their AI agents, controlled by their
hands, face and voice. It runs on their PC. Through it you can see their gestures, whether they are present and
facing the screen, what they say to it, their camera and screen (only when they allow it), and act on the PC
(only when they allow it, while they watch).

How to work with the human through Project Away:
- Check get_context before interrupting: prefer moments when they are present, facing the screen and not
  typing or dictating. ask_user and notify_user accept urgency low|normal|high; low/normal prompts wait for a
  polite moment, high shows at once. Use high only when it really cannot wait.
- Use ask_user for decisions. The person answers with a thumbs up/down, a finger count (option 1-4), a custom
  gesture, a click or their voice. Keep questions short with 2-4 short options.
- Use notify_user for short updates (a toast on their screen, optionally spoken).
- Changing settings: read away://preferences first and cite the preference you are following in `why`. By
  default every change is shown to the person as before/after and needs their thumbs up; low-risk threshold
  changes may apply immediately if they chose "Just do it and tell me". Everything can be undone.
- Using the computer (only with the "use my computer" permission): call request_control and wait for the
  person's approval; act with element names when available (get_screen with elements=true), else coordinates.
  If the person moves their hand, mouse or keyboard you are paused (the takeover pause): stop, and call
  resume_control only if you still need to continue. Risky clicks (send, pay, delete, install...) ask the person first.
- Never ask the person for passwords; you cannot type into password fields and cannot use a shell or files.
- Subscribe to away://events (or call wait_for_event) for: presence (sat down / left), attention, the summon
  gesture or "hey agent" phrases, "that was wrong" feedback, auto-detector trips and control pauses.
"""


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
        "## How to talk to them",
        "- ask_user for decisions (gesture or voice answers), notify_user for updates.",
        "- Wait for a polite moment (present, facing the screen, not typing) unless it is urgent.",
        "- Respect the takeover pause: if they move, you stop.",
        "- Read away://preferences before changing settings and say which preference you followed.",
    ]
    return "\n".join(lines) + "\n"
