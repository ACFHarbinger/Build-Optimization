"""Filesystem-level data file tracking for WSTracker.

:class:`FilesystemTracker` wraps an active :class:`~tracking.core.run.Run`
and provides a higher-level API for recording the full lifecycle of dataset
files on disk: generation, loading, saving, and SHA-256-based change detection.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from tracking.core.run import Run
from tracking.validation.hashing import hash_file


class FilesystemTracker:
    """Tracks data file changes and dataset mutations within a single run.

    Maintains a hash registry for registered files and can detect on-disk
    modifications by comparing the current SHA-256 against the stored value.

    Args:
        run: The active tracking run to associate events with.
    """

    def __init__(self, run: Run) -> None:
        self._run = run
        self._watched: Dict[str, Optional[str]] = {}  # path -> last hash

    # ------------------------------------------------------------------
    # Event methods
    # ------------------------------------------------------------------

    def track_load(
        self,
        file_path: str,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a data file **load** event and register the file for monitoring.

        Args:
            file_path: Path to the loaded dataset file.
            shape: Shape of the loaded dataset.
            metadata: Extra context (e.g. problem type, graph size).
        """
        self._run.log_dataset_event(
            "load",
            file_path=file_path,
            shape=shape,
            metadata={**(metadata or {})},
        )
        if os.path.exists(file_path):
            self._watched[file_path] = hash_file(file_path)

    def track_generate(
        self,
        file_path: Optional[str],
        shape: tuple,
        problem: str,
        graph_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a data **generation** event.

        Args:
            file_path: Output path of the generated file (may be ``None`` for
                in-memory generation).
            shape: Shape of the generated dataset.
            problem: Problem type string (e.g. ``'vrpp'``).
            graph_size: Number of nodes.
            metadata: Extra context dict.
        """
        meta: Dict[str, Any] = {
            "problem": problem,
            "graph_size": graph_size,
        }
        if metadata:
            meta.update(metadata)
        self._run.log_dataset_event(
            "generate",
            file_path=file_path,
            shape=shape,
            metadata=meta,
        )
        if file_path and os.path.exists(file_path):
            self._watched[file_path] = hash_file(file_path)

    def track_mutation(
        self,
        description: str,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an **in-memory mutation** event (e.g. epoch dataset regeneration).

        Args:
            description: Human-readable description of the mutation.
            shape: Shape of the mutated dataset.
            metadata: Extra context dict.
        """
        meta: Dict[str, Any] = {
            "description": description,
        }
        if metadata:
            meta.update(metadata)
        self._run.log_dataset_event(
            "mutate",
            shape=shape,
            metadata=meta,
        )

    def track_save(
        self,
        file_path: str,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a dataset **save** event and update the hash registry.

        Args:
            file_path: Path to the written file.
            shape: Shape of the written data.
            metadata: Extra context dict.
        """
        self._run.log_dataset_event(
            "save",
            file_path=file_path,
            shape=shape,
            metadata={**(metadata or {})},
        )
        if os.path.exists(file_path):
            self._watched[file_path] = hash_file(file_path)

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def watch(self, file_path: str) -> None:
        """Register *file_path* for future change detection without logging."""
        if os.path.exists(file_path):
            self._watched[file_path] = hash_file(file_path)

    def check_changes(self, paths: Optional[List[str]] = None) -> List[str]:
        """Scan watched files (or *paths*) for on-disk hash changes.

        Any file whose hash differs from the registered value is logged as a
        ``hash_change`` event, and the registry is updated.

        Args:
            paths: Specific paths to check.  If ``None``, all watched files
                are checked.

        Returns:
            List of file paths that changed.
        """
        check = paths if paths is not None else list(self._watched.keys())
        changed: List[str] = []
        for fpath in check:
            if not os.path.exists(fpath):
                continue
            current = hash_file(fpath)
            prev = self._watched.get(fpath)
            if prev is not None and current != prev:
                self._run.log_dataset_event(
                    "hash_change",
                    file_path=fpath,
                    metadata={
                        "detected_by": "FilesystemTracker.check_changes",
                        "file_hash": current,
                        "prev_hash": prev,
                        "size_bytes": os.path.getsize(fpath),
                    },
                )
                changed.append(fpath)
            if current is not None:
                self._watched[fpath] = current
        return changed

    def scan_directory(self, directory: str) -> None:
        """Register all files in *directory* for future change detection.

        Files already in the registry are updated; new files are added.

        Args:
            directory: Path to a directory to scan recursively.
        """
        if not os.path.isdir(directory):
            return
        for root, _, files in os.walk(directory):
            for fname in files:
                fpath = os.path.join(root, fname)
                h = hash_file(fpath)
                if h is not None:
                    self._watched[fpath] = h
