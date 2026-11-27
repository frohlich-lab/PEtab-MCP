"""Colours and marks every tab shares, so one kind of thing looks the same on each.

The categorical palette is Okabe and Ito's, which stays apart under the common colour-vision
deficiencies. Every grey here is a middle one, which keeps its contrast on the template's dark theme
as on its light one: the darker greys first chosen vanished on dark.
"""

from __future__ import annotations

#: Okabe-Ito, black last.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")

#: Text and lines that are context, not data.
MUTED = "#8c8c8c"


def categorical(values) -> dict:
    """Each distinct value of ``values``, in order, a palette colour, cycling past eight."""
    seen = list(dict.fromkeys(values))
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(seen)}
