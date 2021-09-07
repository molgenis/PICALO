"""
File:         logger.py
Created:      2020/10/16
Last Changed:
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
    def __init__(self, outdir, clear_log=False):
        self.logfile = os.path.join(outdir, "log.log")

        if clear_log:
            self.clear_logfile()

        self.logger = self.set_logger(self.logfile)

    def clear_logfile(self):
        if os.path.exists(self.logfile) and os.path.isfile(self.logfile):
            os.remove(self.logfile)

    @staticmethod
    def set_logger(logfile):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s "
                   "[%(levelname)-4.4s]  "
                   "%(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(logfile)])

        return logging.getLogger(__name__)

    def get_logger(self):
        return self.logger

    def get_logfile(self):
        return self.logfile