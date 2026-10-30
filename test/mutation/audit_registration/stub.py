#!/usr/bin/env python3
"""Stub profile merger / runner for audit command contract checks."""
import json
import os
from pathlib import Path
import sqlite3
import sys

stage = "merge" if sys.argv[1] == "merge" else "runner"
with open(os.environ["AUDIT_LOG"], "a") as log:
    log.write(json.dumps([stage, sys.argv[1:], os.environ.get("MULL_CONFIG")]) + "\n")
if os.environ.get("FAIL_STAGE") == stage:
    sys.exit(1)
if stage == "merge":
    Path(sys.argv[sys.argv.index("-o") + 1]).touch()
else:
    args = sys.argv
    report = Path(args[args.index("--report-dir") + 1]) / (args[args.index("--report-name") + 1] + ".sqlite")
    with sqlite3.connect(report) as db:
        db.execute("CREATE TABLE mutant (filename TEXT, directory TEXT, execution_status INT, mutator TEXT, line_number INT, column_number INT, mutant_id TEXT)")
        if os.environ.get("FAIL_STAGE") != "validation":
            db.execute("INSERT INTO mutant VALUES (?, '', 1, 'comparison', 1, 1, 'id')", (os.environ["AUDIT_SOURCE"],))
