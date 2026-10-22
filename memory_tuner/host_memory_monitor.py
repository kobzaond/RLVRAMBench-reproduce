"""Sample the current Slurm batch cgroup's host memory without modifying it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time

from memory_tuner.device_contract import write_new_json


def locate(cgroups, job_id, root=Path("/sys/fs/cgroup")):
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("A scheduler job identifier is required")
    for line in cgroups.splitlines():
        _, controllers, relative = line.split(":", 2)
        if not re.search(r"/job[_-]"+job_id+r"(?:[/.]|$)", relative):
            continue
        if ".." in Path(relative).parts:
            raise ValueError("Unexpected cgroup path")
        if controllers == "":
            return root/relative.lstrip("/"), 2
        if "memory" in controllers.split(","):
            return root/"memory"/relative.lstrip("/"), 1
    raise ValueError("Cannot identify this job's memory cgroup")


class HostMemorySampler:
    def __init__(self, attempt, job_id):
        self.attempt, self.job_id = Path(attempt), str(job_id)
        self.stop = threading.Event()
        self.samples = 0
        self.error = None
        self.thread = None
        self.path = None
        self.version = None

    def __enter__(self):
        try:
            self.path, self.version = locate(Path("/proc/self/cgroup").read_text(), self.job_id)
            current = self.path/("memory.current" if self.version==2 else "memory.usage_in_bytes")
            current.read_text()
            self.handle=(self.attempt/"host-memory.jsonl").open("x", buffering=1)
            def sample():
                while not self.stop.is_set():
                    try:
                        total=int(current.read_text())
                        stats=dict(line.split() for line in (self.path/"memory.stat").read_text().splitlines())
                        anonymous=int(stats["anon" if self.version==2 else "total_rss"])
                        cache=int(stats["file" if self.version==2 else "total_cache"])
                        self.handle.write(json.dumps(dict(timestamp_ns=time.time_ns(),current_bytes=total,
                                                          anonymous_bytes=anonymous,file_bytes=cache))+"\n")
                        self.samples+=1
                    except (OSError,ValueError,KeyError) as error:
                        self.error=f"{type(error).__name__}: {error}"
                        break
                    self.stop.wait(1)
            self.thread=threading.Thread(target=sample,daemon=True)
            self.thread.start()
        except (OSError,ValueError) as error:
            self.error=f"{type(error).__name__}: {error}"
        return self

    def __exit__(self, *exception):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
            self.handle.close()
        write_new_json(self.attempt/"host-memory-metadata.json",
                       dict(cgroup=str(self.path) if self.path else None,version=self.version,
                            job_id=self.job_id,pid=os.getpid(),samples=self.samples,error=self.error,
                            scope="Current Slurm batch cgroup including descendant processes and charged file cache; not whole-node memory",
                            interval_seconds=1))
