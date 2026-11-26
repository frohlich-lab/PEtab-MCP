"""A read-only web viewer of one session's version store, for the humans watching or auditing a run.

``petab-mcp-viz --petab-yaml problem.yaml`` serves a HoloViews/Panel app with three tabs over
the store a server writes (:mod:`petab_mcp.provenance`), live or archived:

* **Multistart** (:mod:`.multistart_tab`): how a version's fit went, start by start, and the
  running fit's starts as they finish (:data:`petab_mcp.provenance.FIT_PROGRESS`);
* **Model** (:mod:`.model_tab`): the model as a graph that changes granularity as it is zoomed,
  from agents and their signed influences down to reactions, with the equations of whatever is
  selected (:mod:`.model_math`);
* **Fit** (:mod:`.fit_tab`): the data against the simulation, experiment by experiment with its
  periods, and the residuals.

It is a separate package behind the ``viz`` extra because nothing in it is the server's: the
server never imports it, and it never writes to a store. It reads the store's files directly
(:mod:`.store`) rather than through :class:`~petab_mcp.provenance.ProvenanceStore` or a session,
both of which write as they open. Nothing here compiles a model or simulates one: every number it
shows is one a fit stored.
"""

from __future__ import annotations
