"""Cleans a target Git repository and captures complete per-prompt patches."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class GitRepository:
    """Records the starting revision and manages isolated prompt changes."""

    def __init__(self, directory: str) -> None:
        self.path = Path(directory).resolve()
        root = Path(self.git("rev-parse", "--show-toplevel").strip()).resolve()
        if root != self.path:
            raise ValueError(f"'repo' must be the Git repository root: {root}")
        self.revision = self.git("rev-parse", "--verify", "HEAD").strip()
        self.branch = self.git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def git(self, *arguments: str, index_file: str | None = None) -> str:
        """Runs Git against the target repository and reports command failures."""
        environment = os.environ.copy()
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            environment.pop(key, None)
        if index_file is not None:
            environment["GIT_INDEX_FILE"] = index_file
        try:
            result = subprocess.run(
                ["git", "--no-pager", "-C", str(self.path), *arguments],
                env=environment,
                capture_output=True,
                encoding="utf-8",
                errors="surrogateescape",
                check=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Git {' '.join(arguments)} failed: {error.stderr.strip()}"
            ) from error
        return result.stdout

    def clean(self) -> None:
        """Resets to the starting revision and removes nonignored untracked files."""
        branch = self.git("rev-parse", "--abbrev-ref", "HEAD").strip()
        if branch != self.branch:
            raise RuntimeError("The repository branch changed; refusing to clean another branch.")
        self.git("reset", "--hard", self.revision)
        self.git("clean", "-fd")

    def diff(self) -> str:
        """Returns a binary-capable patch without modifying the real Git index."""
        index = Path(self.git("rev-parse", "--git-path", "index").strip())
        if not index.is_absolute():
            index = self.path / index
        with tempfile.TemporaryDirectory(prefix="locodellm-index-") as directory:
            temporary_index = str(Path(directory) / "index")
            if index.exists():
                shutil.copyfile(index, temporary_index)
            else:
                self.git("read-tree", "HEAD", index_file=temporary_index)
            self.git("add", "--all", "--", ".", index_file=temporary_index)
            return self.git(
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                self.revision,
                "--",
                index_file=temporary_index,
            )
