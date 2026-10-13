"""Attention gate, arm-fatigue mode and break reminders (v0.1.20).

Everything here is pure logic fed with timestamps, so it is unit-tested without a camera:

* :mod:`away.comfort.settings`: the ``attention.*`` / ``comfort.*`` config sections, their validation, their
  Tuning-tab sliders and the plain-language switch labels.
* :mod:`away.comfort.attention`: :class:`~away.comfort.attention.AttentionGate`, "only respond when I'm looking at
  the screen".
* :mod:`away.comfort.fatigue`: :class:`~away.comfort.fatigue.FatigueMonitor` ("switch to head control when my arm is
  tired") and :class:`~away.comfort.fatigue.BreakReminder`.
* :mod:`away.comfort.hub`: :class:`~away.comfort.hub.ComfortHub`, the one object the pipeline talks to.
* :mod:`away.comfort.ui`: indicator colours/labels, tray texts and friendly error messages.
* :mod:`away.comfort.actions`: switch flips and "open the guided first run", shared by the tray and Settings.
"""
