"""Launch the published P-only and Three-P supplementary controls."""

import os
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

from experiments.fedtriad_supplement_c.pipeline import main


if __name__ == '__main__':
    raise SystemExit(main())
