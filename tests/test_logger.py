from src.make_logger import log_fun, make_logger
import logging
from datetime import datetime
from pathlib import Path

LOGGER, FILENAME = make_logger(__file__, test=True)

#! Todo: Improve this module to do all test

@log_fun
def divide(num1, num2):
    return num1 / num2


def test_logfile():
    assert Path(FILENAME).is_file() == True


if __name__ == '__main__':

    LOGGER.debug("Hey")

    result = divide(10, 0)
    print(result)