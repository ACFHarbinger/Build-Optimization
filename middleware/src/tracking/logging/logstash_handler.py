"""
Logstash TCP Handler for remote logging.
"""

import logging
import socket

from .json_formatter import JsonFormatter


class LogstashTcpHandler(logging.Handler):
    """Handler that sends JSON logs over TCP to Logstash."""

    def __init__(self, host: str = "localhost", port: int = 5000):
        """Initialize the TCP handler for Logstash.

        Args:
            host: Logstash server hostname.
            port: Logstash server port.
        """
        super().__init__()
        self.host = host
        self.port = port
        self.formatter = JsonFormatter()

    def emit(self, record):
        """Emit a log record by sending it to Logstash over TCP."""
        try:
            msg = self.format(record) + "\n"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.host, self.port))
                s.sendall(msg.encode("utf-8"))
        except Exception:
            # Silent failure to avoid breaking the application
            pass
