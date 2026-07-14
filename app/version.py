"""Single source of truth for the app version.

The updater fetches this same file from the repo's main branch and compares, so
bumping VERSION here and pushing is what tells existing installs an update is
available. Keep it a plain dotted number so the comparison stays simple.
"""

VERSION = "1.2.0"
