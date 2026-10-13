"""Smart pointer (v0.1.20): magnetic targets, look + say clicking and predictive smoothing.

* :mod:`away.pointer.config`: the ``hands.cursor.magnet`` / ``hands.cursor.predictive`` settings (no app imports).
* :mod:`away.pointer.smoothing`: ``hands.cursor.smoothing: one_euro | predictive`` (speed-adaptive constant-velocity
  Kalman filter with ~1 frame of lookahead) and :func:`~away.pointer.smoothing.make_cursor_filter`.
* :mod:`away.pointer.bench`: the synthetic jitter / latency benchmark that compares the two filters.
* :mod:`away.pointer.targets`: clickable on-screen elements, cached by a low-rate background thread.
* :mod:`away.pointer.uia`: the Windows UI Automation backend (comtypes; free, PyInstaller-friendly).
* :mod:`away.pointer.magnet`: gentle attraction + sticky friction over targets.
* :mod:`away.pointer.looksay`: "click <name>" / "click that" / "double click <name>" / "focus <name>" + numbers.
* :mod:`away.pointer.badges`: numbered badges over ambiguous matches.
* :mod:`away.pointer.smart`: :class:`~away.pointer.smart.SmartPointer`, the one object the pipeline talks to.
* :mod:`away.pointer.settings_ui`: the three plain-language Settings toggles and the friendly status / fix button.

Nothing here imports a GUI toolkit or COM at import time.
"""
