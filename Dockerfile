# The PEtab-MCP server with everything it needs to run: AMICI built, and the toolchain AMICI
# compiles every model version with at run time (a C++ compiler, CMake, Ninja, HDF5, the pip
# SWIG) plus the perl BioNetGen runs under. It is one stage on purpose: the compiler is needed
# by the running server, not only by the install.
#
#   docker build --build-arg PETAB_MCP_REF=$(git rev-parse HEAD) -t petab-mcp .
#   docker run --rm -i -v "$PWD/problem:$PWD/problem" petab-mcp --petab-yaml "$PWD/problem/problem.yaml"
#
# `-i` and never `-t`: stdout is the MCP JSON-RPC channel. On a cluster, convert the image to
# Apptainer (README, "Container").

# Trixie, not bookworm: its GCC 14 matches what AMICI is built with elsewhere (CI's ubuntu-latest
# has 13), where bookworm's GCC 12 is untested. amd64 only, also on Apple silicon (where it runs
# emulated): antimony>=3.2 ships no Linux arm64 wheel, so an arm64 build cannot resolve.
FROM --platform=linux/amd64 python:3.12-slim-trixie

# Deliberately NOT the apt `swig`: compiled models must match the pip `swig` AMICI was built
# against, or they fail to import with "'SwigPyObject' object has no attribute 'module'" (see
# the SWIG steps in .github/workflows/ci.yml). libhdf5-dev with its recommends, as CI installs
# it: AMICI needs HDF5 to send a fit start to a worker process.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build git perl \
    && apt-get install -y libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/petab-mcp
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv "$VIRTUAL_ENV"

# Dependencies first, from pyproject.toml and an empty package, so a source-only change reuses
# this layer instead of rebuilding AMICI (several minutes).
ARG AMICI_PARALLEL_COMPILE=""
COPY pyproject.toml README.md LICENSE /src/
RUN mkdir -p /src/src/petab_mcp && touch /src/src/petab_mcp/__init__.py \
    && AMICI_PARALLEL_COMPILE="${AMICI_PARALLEL_COMPILE:-$(nproc)}" \
       pip install --no-cache-dir /src swig \
    && pip uninstall -y petab-mcp

COPY src /src/src
RUN pip install --no-cache-dir --no-deps /src && rm -rf /src

# Fixed paths for the pip SWIG, resolved as CI's "Pin SWIG" step does. Its bundled binary
# reports a stale -swiglib, so SWIG_LIB must be set too.
RUN python - <<'PY'
import pathlib, swig
data = pathlib.Path(swig.__file__).resolve().parent / "data"
libs = sorted(p for p in (data / "share" / "swig").iterdir() if p.is_dir())
pathlib.Path("/opt/swig/bin").mkdir(parents=True)
pathlib.Path("/opt/swig/bin/swig").symlink_to(data / "bin" / "swig")
pathlib.Path("/opt/swig/lib").symlink_to(libs[-1])
PY
ENV SWIG=/opt/swig/bin/swig SWIG_LIB=/opt/swig/lib

RUN python -c "import amici, pypesto, petab_mcp; print('amici', amici.__version__)"

# The commit the image was built from, for a client to compare with the one it expects.
ARG PETAB_MCP_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/frohlich-lab/PEtab-MCP" \
      org.opencontainers.image.revision="$PETAB_MCP_REF"
RUN echo "$PETAB_MCP_REF" > "$VIRTUAL_ENV/REF"

# `docker run --user` gives a uid with no home; Apptainer sets HOME itself.
ENV HOME=/tmp

ENTRYPOINT ["petab-mcp"]
