"""
File:         logger.py
Created:      2020/10/16
Last Changed: 2021/09/22
Author(s):    M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""
# Standard imports.
import logging
import os

# Third party imports.

# Local application imports.


class Logger:
    def __init__(self, outdir, verbose=False, clear_log=False):
        self.logfile = os.path.join(outdir, "log.log")
        self.verbose = verbose
        self.level = logging.INFO
        if verbose:
            self.level = logging.DEBUG

        if clear_log:
            self.clear_logfile()

        self.logger = self.set_logger()

        self.level_map = {
            0: "NOTSET",
            10: "DEBUG",
            20: "INFO",
            30: "WARNING",
            40: "ERROR",
            50: "CRITICAL"
        }

    def clear_logfile(self):
        if os.path.exists(self.logfile) and os.path.isfile(self.logfile):
            os.remove(self.logfile)

    def set_logger(self):
        # Construct stream handler.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.level)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s  [%(levelname)-4.4s]  %(message)s', "%H:%M:%S"))

        # Construct file handler.
        file_handler = logging.FileHandler(self.logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s  %(module)-16s %(levelname)-8s %(message)s', "%Y-%m-%d %H:%M:%S"))

        # Construct logger object.
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[stream_handler, file_handler])

        return logging.getLogger(__name__)

    def get_logger(self):
        return self.logger

    def print_arguments(self):
        self.logger.info("Arguments:")
        self.logger.info("  > Logfile: {}".format(self.logfile))
        self.logger.info("  > Verbose: {}".format(self.verbose))
        self.logger.info("  > Level: {}".format(self.level_map[self.level]))
        self.logger.info("")