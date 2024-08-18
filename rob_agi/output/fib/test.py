import unittest
from impl import fib

class TestFibonacci(unittest.TestCase):
    def test_fib_zero(self):
        self.assertEqual(fib(0), 0)

    def test_fib_one(self):
        self.assertEqual(fib(1), 1)

    def test_fib_small_numbers(self):
        self.assertEqual(fib(2), 1)
        self.assertEqual(fib(3), 2)
        self.assertEqual(fib(4), 3)
        self.assertEqual(fib(5), 5)

    def test_fib_larger_number(self):
        self.assertEqual(fib(10), 55)

    def test_fib_negative_number(self):
        with self.assertRaises(ValueError):
            fib(-1)

if __name__ == '__main__':
    unittest.main()
