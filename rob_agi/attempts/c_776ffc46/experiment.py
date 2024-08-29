from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_776ffc46.main import solve_776ffc46, visualize_grid

def test_simple_case():
    input_grid = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 2, 0, 3, 5],
        [5, 0, 1, 0, 5],
        [5, 3, 0, 2, 5],
        [5, 5, 5, 5, 5]
    ])
    expected = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 2, 0, 3, 5],
        [5, 0, 2, 0, 5],
        [5, 3, 0, 2, 5],
        [5, 5, 5, 5, 5]
    ])
    actual = solve_776ffc46(input_grid)
    print("Simple case - Input grid:")
    print(visualize_grid(input_grid))
    print("\nSimple case - Actual output:")
    print(visualize_grid(actual))
    print("\nSimple case - Expected output:")
    print(visualize_grid(expected))
    assert actual == expected
