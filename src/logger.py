"""
File:         logger.py
Created:      2020/10/16
Last Changed: 2021/09/22
Author(s):    M.Vochteloo

Copyright (C) 2020 M.Vochteloo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in the LICENSE file in the
root directory of this source tree. If not, see <https://www.gnu.org/licenses/>.
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