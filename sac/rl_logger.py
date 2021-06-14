import datetime
from termcolor import colored


class RLLogger(object):

    def __init__(self, log_file: str = None):
        self.log_file = 'rl.log'
        if log_file:
            self.log_file = log_file
        self.file_pointer = open(self.log_file, 'a')

    def log(self, msg: str):
        self.file_pointer.write(colored(f'{datetime.datetime.now()}:', 'red') + f' {msg}\n')

    def close(self):
        self.file_pointer.close()


if __name__ == '__main__':
    import time
    logger = RLLogger(log_file='test.log')
    i = 1e6
    while i > 0:
        logger.log("test")
        time.sleep(0.5)
        i -= 1
    logger.close()
