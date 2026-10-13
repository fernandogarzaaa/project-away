"""QR codes for the pairing link, as a matrix of booleans (the wizard draws it on a Tk canvas, no image library
needed). Uses the ``qrcode`` package (``[agent]`` extra); returns None without it, and the wizard then shows the
link and the 6-digit code only."""

from __future__ import annotations


def matrix(text: str, border: int = 2) -> list[list[bool]] | None:
    try:
        import qrcode
    except Exception:
        return None
    q = qrcode.QRCode(border=border, error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(text)
    q.make(fit=True)
    return [[bool(c) for c in row] for row in q.get_matrix()]


def ascii_art(text: str) -> str | None:
    """For the terminal (``away agent pair``): two rows per character line."""
    m = matrix(text)
    if m is None:
        return None
    lines = []
    for y in range(0, len(m), 2):
        top, bot = m[y], m[y + 1] if y + 1 < len(m) else [False] * len(m[y])
        lines.append(
            "".join("\u2588" if t and b else "\u2580" if t else "\u2584" if b else " " for t, b in zip(top, bot))
        )
    return "\n".join(lines)
