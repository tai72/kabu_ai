import os
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, ERROR
from logging.handlers import RotatingFileHandler

ROOT_PATH = os.getcwd().replace('src', '')

class Logger:
    def __init__(
        self, 
        program_name
    ):
        self._log_level = 'DEBUG'
        self._program_name = program_name
        self.logger = None

    def initialize_logger(
        self
    ) -> getLogger:
        """ロガーのセッティング"""

        # Instance of 'getLogger'.
        self.logger = getLogger(__name__)
        self.logger.setLevel(self._log_level)

        # Settings of format.
        formatter = Formatter(fmt="[%(levelname)s] (%(asctime)s) %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Settings of 'StreamHandler'.
        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # Settings of fileHandler.
        dt_now = datetime.now()
        file_name = dt_now.strftime(f'[{self._program_name}]%Y%m%d_%H%M%S')
        # file_handler = RotatingFileHandler(filename=os.path.join('log', f"{file_name}.log"), maxBytes=1024 * 1024 * 5, backupCount=30)
        file_handler = RotatingFileHandler(filename=ROOT_PATH + f'log/{file_name}.log', maxBytes=1024 * 1024 * 5, backupCount=30)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self._log_level)
        self.logger.addHandler(file_handler)

        return self.logger
