"""`python -m egemma2 ...` runs the egemma2 binary installed with this package (same as `egemma2 ...`)."""

import subprocess
import sys

from .client import find_egemma2_bin


def main():
    try:
        sys.exit(subprocess.call([find_egemma2_bin(), *sys.argv[1:]]))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
