
import sys
import logging

###### Setup Logging #########
def setUpLogger():
    logger = logging.getLogger(__name__)
    fileHandler = logging.FileHandler('datagen.log') 
    stdoutHandler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(fileHandler)
    logger.addHandler(stdoutHandler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    return logger