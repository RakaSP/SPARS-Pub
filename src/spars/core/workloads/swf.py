"""Convert a Standard Workload Format (CS HUJI Parallel Workloads Archive) log
into a SPARS workload JSON.

usage: python -m spars.core.workloads.swf SOURCE.swf[.gz] OUTPUT.json [LAST_JOB] [FIRST_JOB]

FIRST_JOB and LAST_JOB are 1-based positions in submit order, both inclusive
(default: from the first to the last job).

Mapping (verified to reproduce data/workloads/json/DAS2-fs*-1-3000.json and
SDSC-BLUE-2000-4.2-cln-1-3000.json exactly):

    job_id   <- job number              subtime <- submit time, shifted so the
    res      <- allocated procs, else           first kept job is at 0
               requested procs          reqtime <- requested time, else runtime
    runtime  <- run time                user_id <- user id
    nb_res   <- header MaxProcs, else the largest job

Jobs without a known runtime (-1, e.g. cancelled before start) or without any
processor count are dropped. Everything else in SWF (wait time, memory, queue,
status, dependencies, ...) has no SPARS field and is lost. SWF counts
processors and SPARS nodes, so ``res`` and ``nb_res`` are processors, as in
the existing converted workloads.

A range that starts after job 1 is cut like data/workloads/slice_workload.py:
``subtime`` counts from the submission of the job before the range, and that
shift is stored as ``time_origin``. So jobs 2001-2500 of a trace give the same
file as slicing jobs 2001-2500 out of its 1-3000 conversion.
"""

import gzip
import json
import re
import sys
from pathlib import Path


def convert_swf(source: str | Path, limit: int | None = None, first: int = 1) -> dict:
    """Convert jobs ``first`` to ``limit`` (1-based, inclusive, by submit time) of ``source``.

    ``limit`` None means up to the last job.
    """
    if first < 1 or (limit is not None and limit < first):
        raise ValueError(f"Job range {first}-{limit} is empty: the first job is 1 and the last must not be before the first")
    opener = gzip.open if str(source).endswith(".gz") else open
    max_procs = None
    rows = []
    with opener(source, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(";"):
                match = re.match(r";\s*MaxProcs:\s*(\d+)", line)
                if match:
                    max_procs = int(match.group(1))
            elif line:
                fields = line.split()
                if len(fields) != 18:
                    raise ValueError(f"Expected 18 SWF fields, got {len(fields)}: {line!r}")
                rows.append([int(float(field)) for field in fields])

    jobs = []
    for r in rows:
        res = r[4] if r[4] > 0 else r[7]
        if r[3] < 0 or res <= 0:
            continue
        jobs.append(
            {
                "job_id": r[0],
                "res": res,
                "subtime": r[1],
                "reqtime": r[8] if r[8] > 0 else max(r[3], 1),
                "runtime": r[3],
                "profile": "100",
                "user_id": r[11],
            }
        )
    if not jobs:
        raise ValueError(f"No usable jobs in {source}")

    jobs.sort(key=lambda job: job["subtime"])
    if first > len(jobs):
        raise ValueError(f"{source} has {len(jobs)} usable jobs; the range starts at job {first}")
    jobs = jobs[:limit]
    origin = jobs[0]["subtime"]
    for job in jobs:
        job["subtime"] -= origin
    workload: dict = {}
    if first > 1:
        shift = jobs[first - 2]["subtime"]
        jobs = jobs[first - 1:]
        for job in jobs:
            job["subtime"] -= shift
        workload["time_origin"] = shift
    nb_res = max_procs or max(job["res"] for job in jobs)
    return {"nb_res": max(nb_res, max(job["res"] for job in jobs)), "jobs": jobs, **workload}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python -m spars.core.workloads.swf SOURCE.swf OUTPUT.json [LAST_JOB] [FIRST_JOB]")
    workload = convert_swf(
        sys.argv[1],
        int(sys.argv[3]) if len(sys.argv) > 3 else None,
        int(sys.argv[4]) if len(sys.argv) > 4 else 1,
    )
    with open(sys.argv[2], "w", encoding="utf-8") as stream:
        json.dump(workload, stream, indent=4)
        stream.write("\n")
    print(f"{len(workload['jobs'])} jobs, nb_res={workload['nb_res']}")
