"""Allow ``python -m acdb ...`` in addition to the ``acdb`` console script."""

from __future__ import annotations

from acdb.cli import main

if __name__ == "__main__":
    main()
