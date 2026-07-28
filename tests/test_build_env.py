"""How a failed model BUILD is reported.

Nothing here touches the environment: AMICI's own conda/PATH handling is left alone. These pin
only the MESSAGE — that a build failure is not reported to the agent as a broken model, and that
the compiler's own diagnostic reaches whoever reads the rejection.
"""

from __future__ import annotations

import pytest


# --- a build failure must not be reported as a modelling failure -----------------------------
@pytest.mark.parametrize("message", [
    "amici/model.h:14:10: fatal error: span: No such file or directory",
    "Command '['cmake', '--build', '...']' returned non-zero exit status 1.",
    "Failed building the model extension",
    "swig: command not found",
    # NOT included: a bare "[Errno 28] No space left on device" / "Permission denied". Those were
    # matched by the first version anywhere in the text, which caught fit-time OSErrors and told
    # the agent not to re-author the model that WAS the problem. A disk-full during a BUILD still
    # carries build context (setup.py / cmake) and is caught by those tokens instead.
])
def test_a_toolchain_failure_is_not_blamed_on_the_model(message):
    """The agent's only lever is ``set_model``, so telling it a BUILD failure "usually means it is
    ill-posed for estimation" sends it to rewrite a model that was already correct, and the real
    cause on the host never gets looked at. The C++20 case cost exactly that misdiagnosis twice."""
    from petab_mcp.operations import _compile_fit_failure_message

    out = _compile_fit_failure_message(RuntimeError(message))
    assert "toolchain problem" in out
    assert "ill-posed" not in out
    assert "Do not re-author" in out


def test_a_genuine_model_failure_still_gets_the_ill_posed_hint():
    """The toolchain branch must not swallow the case the message was written for."""
    from petab_mcp.operations import _compile_fit_failure_message

    out = _compile_fit_failure_message(RuntimeError("CVODES failed with flag -4 at t = 3.2"))
    assert "ill-posed" in out and "toolchain problem" not in out


def test_a_reserved_name_clash_still_wins_over_the_toolchain_branch():
    """An AMICI reserved-symbol clash surfaces AS a build error, but IS a model problem and has a
    precise fix (rename), so it must be matched before the generic toolchain check."""
    from petab_mcp.operations import _compile_fit_failure_message

    out = _compile_fit_failure_message(KeyError("amici_h"))
    assert "collides" in out and "toolchain problem" not in out


def test_the_real_amici_build_failure_is_recognised():
    """The matcher's whole point. AMICI catches the inner cmake error, PRINTS the diagnostics, and
    re-raises the OUTER CalledProcessError, whose text carries no compiler wording at all — so the
    first version of this matched nothing in production while its test passed against a fabricated
    'fatal error: span' string. Both the real text AND the real type must be recognised."""
    import subprocess

    from petab_mcp.operations import _compile_fit_failure_message

    real_text = ("Command '['/x/bin/python', PosixPath('/tmp/b/setup.py'), '--verbose', "
                 "'build_ext', '--build-lib=/tmp/b']' returned non-zero exit status 1.")
    for exc in (RuntimeError(real_text),
                subprocess.CalledProcessError(1, ["python", "setup.py", "build_ext"])):
        out = _compile_fit_failure_message(exc)
        assert "toolchain problem" in out and "ill-posed" not in out, out
def test_a_build_failure_carries_what_the_compiler_actually_said():
    """AMICI runs the build with stdout=PIPE/stderr=STDOUT and, on failure, prints `e.output` and
    re-raises — so `str(exc)` is only the wrapper (`Command '[...]' returned non-zero exit status
    1.`) while the real cause sits on the exception and reached nobody. Diagnosing a red CI from
    the wrapper alone cost several rounds; the message must carry the diagnostic itself."""
    import subprocess

    from petab_mcp.operations import _compile_fit_failure_message

    exc = subprocess.CalledProcessError(
        1, ["python", "setup.py", "build_ext"],
        output=b"[1/9] building x.cpp\nmodel.h:14:10: fatal error: span: No such file "
               b"or directory\ncompilation terminated.\n")
    out = _compile_fit_failure_message(exc)
    assert "fatal error: span" in out
    assert "What the build actually reported" in out


def test_the_captured_output_alone_can_classify_a_toolchain_failure():
    """The wrapper text may carry no build vocabulary at all (that is exactly the outer
    CalledProcessError AMICI re-raises). If the CAPTURED output names the real cause, that is
    enough to stop blaming the model."""
    from petab_mcp.operations import _compile_fit_failure_message

    class _Weird(Exception):
        output = "gcc: error: unrecognized command line option '-std=c++20'\n"

    out = _compile_fit_failure_message(_Weird("something opaque"))
    assert "toolchain problem" in out and "ill-posed" not in out


def test_a_model_failure_with_no_build_output_is_unaffected():
    """No captured output must mean no appendix and no reclassification."""
    from petab_mcp.operations import _compile_fit_failure_message

    out = _compile_fit_failure_message(RuntimeError("CVODES failed with flag -4 at t = 3.2"))
    assert "ill-posed" in out and "What the build actually reported" not in out
