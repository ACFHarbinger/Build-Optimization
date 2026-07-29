"""Logger redirection to file for simulation output.

Provides LoggerWriter to tee stdout/stderr to both terminal and file,
filtering tqdm progress bars from file output.
"""

import os
import re
import sys
from datetime import datetime
from typing import Optional

import hydra


class LoggerWriter:
    """
    Fake file-like stream object that redirects writes to both terminal and file.
    """

    def __init__(self, terminal, filename, echo_to_terminal=True):
        """Initialize the logger writer.

        Args:
            terminal: Original terminal stream (stdout or stderr).
            filename: Path to the log file.
            echo_to_terminal: If True, also write to terminal.
        """
        self.terminal = terminal
        self.filename = filename
        self.echo_to_terminal = echo_to_terminal
        # Open file in append mode.
        self.log = open(filename, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, message):
        """Write message to terminal and/or log file, filtering based on tags line-by-line."""
        if self.echo_to_terminal:
            self.terminal.write(message)

        # Logic to filter based on tags: [INFO], [WARNING], [ERROR]
        # Pattern matches:
        # ^                 : Start of string
        # (\x1b\[[0-9;]*m)* : Zero or more ANSI escape sequences
        # \s*               : Optional whitespace
        # \[(INFO|WARNING|ERROR)\] : The tag
        tag_pattern = re.compile(r"^(\x1b\[[0-9;]*m)*\s*\[(INFO|WARNING|ERROR|DEBUG|CRITICAL)\]")

        # Pattern to match ANSI escape sequences (colors, styles, cursor control)
        # matches \x1b[ (start), then 0-9; (params), then m, K, H, F (codes)
        ansi_escape = re.compile(r"\x1b\[[0-9;]*[mKHF]")

        # splitlines(True) keeps the line endings (\n)
        for line in message.splitlines(keepends=True):
            if tag_pattern.match(line) or ("\r" not in line and len(line.strip()) > 0):
                # Strip ANSI codes before writing to log file
                clean_line = ansi_escape.sub("", line)
                self.log.write(clean_line)

        self.log.flush()  # Ensure it writes immediately

    def flush(self):
        """Flush both terminal and log file buffers."""
        if self.echo_to_terminal:
            self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file handle."""
        self.log.close()

    def isatty(self):
        """Return whether the terminal is a TTY."""
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self):
        """Return the file descriptor number of the terminal."""
        return getattr(self.terminal, "fileno", lambda: -1)()

    def __getattr__(self, name):
        """Delegate any other attribute access to the terminal stream."""
        return getattr(self.terminal, name)


def setup_logger_redirection(
    log_file: Optional[str] = None,
    silent: bool = False,
    redirect_stdout: bool = True,
    redirect_stderr: bool = True,
    echo_to_terminal: bool = False,
) -> str:
    """Redirects stdout and stderr to a timestamped log file.

    Args:
        log_file: Optional path to an existing log file to reuse.
        silent: If True, do not print the redirection announcement.
        redirect_stdout: If True, redirect sys.stdout.
        redirect_stderr: If True, redirect sys.stderr.
        echo_to_terminal: If True, keep echoing to the original terminal.
    """
    if log_file is None:
        try:
            base_dir = hydra.utils.get_original_cwd()
        except (ValueError, ImportError):
            base_dir = os.getcwd()

        log_dir = os.path.join(base_dir, "logs/simulations")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{timestamp}.log")

    # Tee stdout and stderr based on flags
    if redirect_stdout:
        sys.stdout = LoggerWriter(sys.stdout, log_file, echo_to_terminal=echo_to_terminal)
    if redirect_stderr:
        sys.stderr = LoggerWriter(sys.stderr, log_file, echo_to_terminal=echo_to_terminal)

    # Reconfigure Loguru to use the redirected sys.stderr
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, format="[{level}] {message}")
    except ImportError:
        pass

    # Print to ORIGINAL stderr so user knows where logs are, before redirection takes full effect
    if not silent:
        if hasattr(sys.stderr, "terminal"):
            sys.stderr.terminal.write(f"Logging simulation output to: {log_file}\n")
        else:
            sys.stderr.write(f"Logging simulation output to: {log_file}\n")

    return log_file
