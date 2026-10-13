"""Owned subprocess transport for the READY/RESULT/ERROR/BYE protocol."""

import json
import os
import subprocess
import sys
import threading


class Worker:
    def __init__(self, name, config, event_queue):
        self.name, self.config, self.event_queue = name, config, event_queue
        self.process = None

    def start(self):
        env = {**os.environ, **self.config.get("env", {}), "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen(
            self.config["command"],
            cwd=self.config.get("cwd"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.process.stdout:
            line = line.rstrip()
            kind, separator, data = line.partition("|")
            if separator and kind in {"READY", "RESULT", "ERROR", "BYE"}:
                try:
                    self.event_queue.put((self.name, kind, json.loads(data)))
                except (ValueError, TypeError) as exc:
                    self.event_queue.put(
                        (
                            self.name,
                            "ERROR",
                            {"error": f"Malformed protocol message: {exc}"},
                        )
                    )
            else:
                print(f"[{self.name}] {line}", file=sys.stderr)

    def send(self, payload):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"Worker {self.name} is not running")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def shutdown(self):
        if self.process and self.process.poll() is None:
            try:
                self.send({"op": "shutdown"})
            except (OSError, RuntimeError):
                pass
