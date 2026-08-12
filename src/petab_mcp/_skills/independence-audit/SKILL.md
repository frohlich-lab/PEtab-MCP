---
name: independence-audit
description: >-
  Read when residuals stay structured after re-fitting. Which independence assumption the data is
  denying, the ways a model can meet it, and how to derive the check yourself when the diagnostics
  cannot answer.
---

# Which independence assumption is the data denying?

A misfit that survives re-fitting is usually an **independence assumption** your model imposes that
the data denies — states acting independently, one relaxation timescale, a linear input, no
conserved total. Re-fitting cannot fix that, and neither can parameters that leave the assumption
intact. You need the missing mechanism, and this skill is about identifying WHICH one.

Start with the regime diagnostics: they read the measurements directly and report what any model
must reproduce — whether the response rises and falls, whether a second timescale is
distinguishable, how steep the dose-response is, whether a total is conserved. They need no fit and
no model, so read them before theorising. A structured residual tells you something is missing; the
diagnostics tell you what.

Read them in one direction only. A non-null `requirement` is ESTABLISHED and forces that structure
in; `support_for_extension: suggestive` leans that way without establishing it, which is a reason to
try it. A null `requirement` — like `testable: false` — says only that this DESIGN did not settle
the question, which is the ordinary outcome for a test with little power. It is never evidence that
the simpler form suffices, so it neither answers the audit below nor excuses skipping it: where the
data are silent, the assumption still has to be judged on the mechanism.

## Ways to meet a requirement

Most requirements have several encodings, and they differ in what they ASSUME about the system:

| requirement | encodings that can produce it |
|---|---|
| response rises then falls | deplete an upstream pool; feedback from a state you have; a new intermediate species |
| a second timescale | make an existing rate saturating; split an existing flux; a new buffer state |
| dose-response steeper than one binding step | a cooperativity exponent; saturation on an existing rate; a multi-step cascade |
| effects are not separable | an interaction term on states already present; a named complex per combination |
| a total is conserved | impose it as a conservation relation |

Pick what the biology and the diagnostics support. Where two are equally supported, fit each and let
the score say which the data prefers — that is a measurement, and it beats guessing in advance.

## When the data cannot tell you

A `testable: false` verdict means the DESIGN cannot answer that question — not that the answer is
no. It is common: a dose-response slope needs several dose levels, a conserved total needs every
observable measured together, and a shape verdict needs enough time points. Steady-state and
combinatorial problems may return almost nothing.

There, fall back to deriving the check yourself, and note that as the lead you cannot read the
measurement table — ask an analyst for a NUMBER, not a characterisation:

> "From the measured data only, not the simulation: compute <quantity> per <group>. Return the value
>  and its spread — across replicates where the design has them, else the successive-difference
>  scatter, saying which. Nothing else."

A vague ask returns prose you cannot compare against a prediction; a numeric ask returns a number.
Pick a quantity your current factorisation forces to a particular value *whatever the parameters
are* — a ratio, a slope bound, an invariance — so the comparison tests the structure rather than the
fit.

## Reading the residual evidence

- "No significant structure" at **low power** means *not tested*. Treat it as unknown, not clean.
- Structured residuals **with** a comfortable goodness-of-fit is the strongest signal that a
  factorisation is wrong: the noise model is absorbing a shape it should not have to.
- Per-observable bias and worst-fit points tell you WHICH observable to think about. They do not
  tell you which parameter to retune — that is the trap this skill exists to avoid.
- When sigmas are estimated, goodness-of-fit is close to uninformative by construction; the residual
  STRUCTURE remains the honest test.
