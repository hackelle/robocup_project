#!/usr/bin/env python2

import logging
from threading import Lock


class GUILogger(logging.Handler):
    """Logger that originally logged to the GUI, but this no longer works."""

    widget = None
    mutex = Lock()

    def __init__(self, widget=None):
        super(GUILogger, self).__init__()

        self.setFormatter(logging.Formatter(
            "[%(levelname)s]\t[%(asctime)s]: %(message)s"
        ))

        if widget is not None:
            GUILogger.widget = widget

    def emit(self, record):
        # with GUILogger.mutex:
        msg = self.format(record)
        print(msg)
        # GUILogger.widget.appendPlainText(msg)

    def write(self, m):
        pass
