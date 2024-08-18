def fib(n):
    """
    Calculate the nth Fibonacci number.
    
    :param n: The position of the Fibonacci number to calculate (non-negative integer)
    :return: The nth Fibonacci number
    :raises ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
