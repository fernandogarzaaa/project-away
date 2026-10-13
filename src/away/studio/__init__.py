"""Gesture Studio (v0.1.20): zones, per-class model visibility, custom gestures from a blank model, packs.

Everything here except :mod:`away.studio.ui` is plain logic (numpy, no Tk, no camera) so it is unit-tested
headless. The Settings tab lives in :mod:`away.studio.ui`; the camera worker in :mod:`away.studio.live`; the
runtime gate the app uses in :mod:`away.studio.engine`.
"""
