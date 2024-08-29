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

def test_edge_cases():
    # Test case 1: Only red adjacent to gray
    input_grid1 = ColoredGrid(values=[
        [5, 5, 5],
        [5, 1, 5],
        [5, 2, 5]
    ])
    print("\nTest case 1 - Only red adjacent to gray:")
    print(visualize_grid(solve_776ffc46(input_grid1)))

    # Test case 2: Only green adjacent to gray
    input_grid2 = ColoredGrid(values=[
        [5, 5, 5],
        [5, 1, 5],
        [5, 3, 5]
    ])
    print("\nTest case 2 - Only green adjacent to gray:")
    print(visualize_grid(solve_776ffc46(input_grid2)))

    # Test case 3: Equal red and green adjacent to gray
    input_grid3 = ColoredGrid(values=[
        [5, 5, 5, 5],
        [5, 1, 2, 5],
        [5, 3, 1, 5],
        [5, 5, 5, 5]
    ])
    print("\nTest case 3 - Equal red and green adjacent to gray:")
    print(visualize_grid(solve_776ffc46(input_grid3)))

    # Test case 4: Multiple blue plus shapes
    input_grid4 = ColoredGrid(values=[
        [5, 5, 5, 5, 5, 5, 5],
        [5, 1, 0, 1, 0, 2, 5],
        [5, 0, 1, 0, 1, 3, 5],
        [5, 1, 0, 1, 0, 0, 5],
        [5, 5, 5, 5, 5, 5, 5]
    ])
    print("\nTest case 4 - Multiple blue plus shapes:")
    print(visualize_grid(solve_776ffc46(input_grid4)))

if __name__ == "__main__":
    test_simple_case()
    test_edge_cases()
