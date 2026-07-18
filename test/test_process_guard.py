import queue
import unittest

from aiphy.main import run_method_in_process


class SimulatedPyo3Panic(BaseException):
    pass


class BrokenWorker:
    @staticmethod
    def fail():
        raise SimulatedPyo3Panic("rust getter panic")


class ChildProcessGuardTests(unittest.TestCase):
    def test_baseexception_is_reported_to_parent_queue(self):
        result_queue = queue.Queue()
        shared = []

        run_method_in_process(BrokenWorker(), "fail", result_queue, shared)

        self.assertEqual(result_queue.get_nowait(), [(None, 0)])


if __name__ == "__main__":
    unittest.main()
