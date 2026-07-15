import os

# Cap native thread-pool sizes *before* numpy/numba/datashader (imported by the submodules
# below) get a chance to launch their own pools -- on a high-core-count machine, numba in
# particular launches a persistent OS thread pool sized to os.cpu_count() the first time a
# parallel-JIT function runs (e.g. inside datashader's regrid path, used whenever
# use_datashader=True), and that pool cannot be resized afterward: calling
# numba.set_num_threads() or reassigning the env var later raises RuntimeError once numba's
# threads are already launched, or is simply too late to matter. On a 120-core box this alone
# produced 500+ extra OS threads from a single annotator(use_datashader=True) render (multiplied
# further by file-backed mode's per-chunk regrid calls), which manifests as the Jupyter kernel
# dying with no traceback (thread/memory exhaustion, not a Python exception) once a few pan/zoom
# navigations pile the cost up. Using setdefault() so an explicit choice made by the user (or
# already set by whatever imported numba/BLAS first) is never overridden.
for _env_var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env_var, str(min(4, os.cpu_count() or 1)))
del _env_var

# On macOS in particular, numba's default threading layer (TBB or OpenMP, depending on what's
# installed) can conflict with Apple's own Accelerate/vecLib BLAS backend -- which numpy
# typically uses on Apple Silicon -- when both try to manage native threads in the same process.
# This is a separate, well-documented crash class from the thread-*count* issue above (it doesn't
# show up as runaway thread counts; it can hang or crash outright, silently, regardless of how few
# threads are requested). numba's own docs recommend 'workqueue' -- its simplest threading layer,
# with no external OpenMP/TBB dependency -- as the standard fix. setdefault() so this never
# overrides an explicit choice.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from .organaxis import *
from .io import *
from . import annotation
from . import legacy
