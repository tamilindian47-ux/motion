COPY runner.py /app/runner.py
COPY handler.py /app/handler.py

# ------------------------------------------------------------
# Prevent upstream ReHiFace-S argument parsers from reading
# the worker runner's command-line flags.
# ------------------------------------------------------------

RUN python - <<'PY'
from pathlib import Path
import re

path = Path("/app/runner.py")
source = path.read_text()

marker = "# REHIFACE_ARGV_ISOLATION"

if marker in source:
    print("Runner argument-isolation patch already present.", flush=True)
else:
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)run\(arguments\)[ \t]*$",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(source))

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one standalone run(arguments) call "
            f"in runner.py; found {len(matches)}. "
            "No changes were made."
        )

    match = matches[0]
    indent = match.group("indent")

    replacement = "\n".join([
        indent + marker,
        indent + "import sys",
        indent + "_rehiface_original_argv = sys.argv[:]",
        indent + "try:",
        indent + "    sys.argv = [sys.argv[0]]",
        indent + "    run(arguments)",
        indent + "finally:",
        indent + "    sys.argv = _rehiface_original_argv",
    ])

    patched = (
        source[:match.start()]
        + replacement
        + source[match.end():]
    )

    compile(patched, str(path), "exec")
    path.write_text(patched)

    print("Runner argument-isolation patch applied.", flush=True)
PY
