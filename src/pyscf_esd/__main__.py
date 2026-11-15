"""Support python -m pyscf_esd as well as the installed console script."""

from .cli import main

raise SystemExit(main())
