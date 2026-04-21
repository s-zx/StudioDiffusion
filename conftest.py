# Marker file: makes pytest treat the repo root as rootdir, so test imports
# resolve `adapters`, `segmentation`, etc. without needing PYTHONPATH or
# `pip install -e .`. Required because pyproject.toml has a broken
# `build-backend` value ("setuptools.backends.legacy:build" is not a real
# entry point) — fixing that is out of scope for the IP-Adapter work.
