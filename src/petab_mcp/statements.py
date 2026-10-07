"""The mechanism vocabulary: what one model edit ASSERTS.

This module owns the layer the paper's Appendix B.3 calls bindings (``S``) and records as not
currently separable — today a binding exists only as a reference inside the authored model
text, so nothing can be revised across its instances. Here it is declared by the agent at edit
time and stored beside the model, never recovered from it: inference from finished rate laws is
pattern matching run backwards, and the model does not record which species are modified forms
of which.

Statements are ``indra.statements.Statement`` instances. Nearly all are INDRA's own —
Phosphorylation, Complex, Activation, IncreaseAmount, Translocation and the rest cover what a
molecular model asserts. The one type here covers what INDRA's vocabulary does not, and the gap is
not an oversight on INDRA's part: its types describe molecular biochemistry, while a third of the
benchmark collection is cell-population and epidemiological dynamics. Confining the agent to
INDRA's set would exclude those benchmarks outright.

It is written as an INDRA subclass rather than as a parallel vocabulary so that one dispatch
serves both: the PySB assembler, Agent grounding, ``Preassembler`` deduplication and belief all
key on the base class, not on membership of INDRA's own set. What that does not buy is INDRA's web
services and network viewers, which enumerate the built-in types; nothing on the fitting or
scoring path depends on them.

Grounding is OPTIONAL here, deliberately. A molecular agent carries ``db_refs`` and stays
comparable against the corpus; a T-cell compartment carries none, and neither does an unnamed
mediator — which needs no type of its own, being an ungrounded agent in an ordinary statement.
An absent grounding warns rather than rejects (:func:`ungrounded`).
"""

from __future__ import annotations

from indra.statements import Agent, Statement
from indra.statements import statements as _indra_statements
from indra.statements.statements import mk_str


class PopulationTransition(Statement):
    """One population becomes another: naive to effector, susceptible to infected.

    Not INDRA's ``Conversion``, which converts molecular species with explicit reactant and
    product lists and carries a catalyst slot that has no reading here.
    """

    _agent_order = ["source", "target"]

    def __init__(self, source: Agent, target: Agent, evidence=None) -> None:
        super().__init__(evidence)
        self.source, self.target = source, target

    def matches_key(self) -> str:
        return mk_str((type(self).__name__, *(a.matches_key() for a in self.agent_list())))

    def set_agent_list(self, agent_list: list[Agent]) -> None:
        for name, agent in zip(self._agent_order, agent_list, strict=True):
            setattr(self, name, agent)

    def to_json(self, use_sbo: bool = False, matches_fun=None) -> dict:
        json_dict = super().to_json(use_sbo, matches_fun)
        for name in self._agent_order:
            json_dict[name] = getattr(self, name).to_json()
        return json_dict

    @classmethod
    def _from_json(cls, json_dict: dict) -> PopulationTransition:
        return cls(*(Agent._from_json(json_dict[name]) for name in cls._agent_order))

    def __str__(self) -> str:
        return f"{type(self).__name__}({', '.join(str(a) for a in self.agent_list())})"


# INDRA deserialises by looking the ``type`` field up in its OWN module, and ``stmts_from_json``
# DROPS a statement it cannot resolve with only a logged warning — so without this a stored
# PopulationTransition vanishes on restore rather than failing.
setattr(_indra_statements, PopulationTransition.__name__, PopulationTransition)


def ungrounded(stmt: Statement) -> list[str]:
    """Agent names in ``stmt`` carrying no ``db_refs``, for the warning path.

    ``TEXT`` does not count: it is the string a reader saw, not an identity.
    """
    return [a.name for a in stmt.real_agent_list() if not set(a.db_refs) - {"TEXT"}]
