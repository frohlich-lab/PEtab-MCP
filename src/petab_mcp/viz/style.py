"""Colours and marks every tab shares, so one kind of thing looks the same on each.

The categorical palette is Okabe and Ito's, which stays apart under the common colour-vision
deficiencies; an influence's sign is therefore blue against vermillion, never green against red.
"""

from __future__ import annotations

#: Okabe-Ito, black last.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")

#: Text and lines that are context, not data.
MUTED = "#8c8c8c"

#: An edge kind's colour and dash (:data:`petab_mcp.viz.model_math.EDGE_KINDS` and the reaction
#: level's ``consumes``/``produces``).
EDGES: dict[str, tuple[str, str]] = {
    "activates": ("#0072B2", "solid"),
    "inhibits": ("#D55E00", "solid"),
    "modulates": ("#8c8c8c", "dashed"),
    "modifies": ("#4d4d4d", "solid"),
    "converts": ("#009E73", "solid"),
    "binds": ("#CC79A7", "solid"),
    "role": ("#666666", "solid"),
    "reads": ("#b3b3b3", "dotted"),
    "consumes": ("#666666", "solid"),
    "produces": ("#666666", "solid"),
}

#: A node kind's marker, colour and size.
NODES: dict[str, tuple[str, str, int]] = {
    "agent": ("circle", "#56B4E9", 22),
    "species": ("circle", "#56B4E9", 14),
    "row": ("square", "#E69F00", 14),
    "reaction": ("square", "#4d4d4d", 7),
    "observable": ("diamond", "#CC79A7", 14),
}

#: Added since the compared version, and present only there.
ADDED = "#009E73"
REMOVED = "#bdbdbd"


def categorical(values) -> dict:
    """Each distinct value of ``values``, in order, a palette colour, cycling past eight."""
    seen = list(dict.fromkeys(values))
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(seen)}
