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

from .organaxis import *
from .io import *
from . import annotation
from . import legacy
