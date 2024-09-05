from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_776ffc46.main import solve_776ffc46

def visualize_grid(grid):
    color_map = {0: '⬛', 1: '🟦', 2: '🟥', 3: '🟩', 4: '🟨', 5: '⬜'}
    return '\n'.join(''.join(color_map.get(cell, '❓') for cell in row) for row in grid.values)

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
    assert actual == expected, f"Expected:\n{visualize_grid(expected)}\n\nActual:\n{visualize_grid(actual)}"

def test_edge_cases():
    # Test case 1: Only red adjacent to gray
    input_grid1 = ColoredGrid(values=[
        [5, 5, 5],
        [5, 1, 5],
        [5, 2, 5]
    ])
    print("\nTest case 1 - Only red adjacent to gray:")
    result1 = solve_776ffc46(input_grid1)
    print(visualize_grid(result1))
    assert result1.values[1][1] == 2, f"Expected red (2), but got {result1.values[1][1]}"

    # Test case 2: Only green adjacent to gray
    input_grid2 = ColoredGrid(values=[
        [5, 5, 5],
        [5, 1, 5],
        [5, 3, 5]
    ])
    print("\nTest case 2 - Only green adjacent to gray:")
    result2 = solve_776ffc46(input_grid2)
    print(visualize_grid(result2))
    assert result2.values[1][1] == 3, f"Expected green (3), but got {result2.values[1][1]}"

    # Test case 3: Equal red and green adjacent to gray
    input_grid3 = ColoredGrid(values=[
        [5, 5, 5, 5],
        [5, 1, 2, 5],
        [5, 3, 1, 5],
        [5, 5, 5, 5]
    ])
    print("\nTest case 3 - Equal red and green adjacent to gray:")
    result3 = solve_776ffc46(input_grid3)
    print(visualize_grid(result3))
    assert result3.values[1][1] == 2 and result3.values[2][2] == 2, f"Expected red (2) for both plus shapes, but got {result3.values[1][1]} and {result3.values[2][2]}"

    # Test case 4: Multiple blue plus shapes
    input_grid4 = ColoredGrid(values=[
        [5, 5, 5, 5, 5, 5, 5],
        [5, 1, 0, 1, 0, 2, 5],
        [5, 0, 1, 0, 1, 3, 5],
        [5, 1, 0, 1, 0, 0, 5],
        [5, 5, 5, 5, 5, 5, 5]
    ])
    print("\nTest case 4 - Multiple blue plus shapes:")
    result4 = solve_776ffc46(input_grid4)
    print(visualize_grid(result4))
    assert result4.values[1][1] == 2 and result4.values[1][3] == 2 and result4.values[2][4] == 2, f"Expected all plus shapes to be red (2), but got {result4.values[1][1]}, {result4.values[1][3]}, and {result4.values[2][4]}"

    # Test case 5: Complex grid with multiple gray borders
    input_grid5 = ColoredGrid(values=[
        [5, 5, 5, 5, 5, 0, 0, 0],
        [5, 1, 0, 2, 5, 0, 3, 0],
        [5, 0, 1, 0, 5, 0, 0, 0],
        [5, 3, 0, 1, 5, 0, 2, 0],
        [5, 5, 5, 5, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 5, 5],
        [0, 1, 0, 2, 0, 5, 1, 5],
        [0, 0, 1, 0, 0, 5, 5, 5]
    ])
    print("\nTest case 5 - Complex grid with multiple gray borders:")
    result5 = solve_776ffc46(input_grid5)
    print(visualize_grid(result5))
    assert result5.values[2][2] == 2 and result5.values[7][2] == 2, f"Expected red (2) for both plus shapes, but got {result5.values[2][2]} and {result5.values[7][2]}"

    # Test case 6: Diagonal adjacency
    input_grid6 = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 2, 0, 3, 5],
        [5, 0, 1, 0, 5],
        [5, 3, 0, 2, 5],
        [5, 5, 5, 5, 5]
    ])
    print("\nTest case 6 - Diagonal adjacency:")
    result6 = solve_776ffc46(input_grid6)
    print(visualize_grid(result6))
    assert result6.values[2][2] == 2, f"Expected red (2), but got {result6.values[2][2]}"

    # Test case 7: No red or green adjacent to gray
    input_grid7 = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 0, 1, 0, 5],
        [5, 1, 1, 1, 5],
        [5, 0, 1, 0, 5],
        [5, 5, 5, 5, 5]
    ])
    print("\nTest case 7 - No red or green adjacent to gray:")
    result7 = solve_776ffc46(input_grid7)
    print(visualize_grid(result7))
    assert result7 == input_grid7, f"Expected no change, but got:\n{visualize_grid(result7)}"

def test_simple_border_case():
    input_grid = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 1, 0, 5],
        [5, 2, 0, 3, 5],
        [5, 5, 5, 5, 5]
    ])
    expected = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 2, 0, 5],
        [5, 2, 0, 3, 5],
        [5, 5, 5, 5, 5]
    ])
    print("\nSimple border case - Input grid:")
    print(visualize_grid(input_grid))
    actual = solve_776ffc46(input_grid)
    print("\nSimple border case - Actual output:")
    print(visualize_grid(actual))
    print("\nSimple border case - Expected output:")
    print(visualize_grid(expected))
    assert actual == expected, f"Expected:\n{visualize_grid(expected)}\n\nActual:\n{visualize_grid(actual)}"

if __name__ == "__main__":
    test_simple_case()
    test_edge_cases()
    test_simple_border_case()

    # New experiment: Test a simple case with clear border and adjacent colors
    print("\nSimple border experiment:")
    simple_input = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 2, 0, 3, 5],
        [5, 0, 1, 0, 5],
        [5, 3, 0, 2, 5],
        [5, 5, 5, 5, 5]
    ])
    print("Input grid:")
    print(visualize_grid(simple_input))
    simple_result = solve_776ffc46(simple_input)
    print("\nResult grid:")
    print(visualize_grid(simple_result))

    # New experiment: Test multiple blue plus shapes
    print("\nMultiple blue plus shapes experiment:")
    multiple_plus_input = ColoredGrid(values=[
        [5, 5, 5, 5, 5, 5, 5],
        [5, 2, 0, 0, 0, 3, 5],
        [5, 0, 1, 0, 1, 0, 5],
        [5, 0, 0, 1, 0, 0, 5],
        [5, 0, 1, 0, 1, 0, 5],
        [5, 3, 0, 0, 0, 2, 5],
        [5, 5, 5, 5, 5, 5, 5]
    ])
    print("Input grid:")
    print(visualize_grid(multiple_plus_input))
    multiple_plus_result = solve_776ffc46(multiple_plus_input)
    print("\nResult grid:")
    print(visualize_grid(multiple_plus_result))

    # Verify the transformation
    expected_color = 2  # Red, since it's the most frequent adjacent to gray
    center_coords = [(2, 2), (2, 4)]
    for r, c in center_coords:
        assert multiple_plus_result.values[r][c] == expected_color, f"Expected color {expected_color} at ({r}, {c}), but got {multiple_plus_result.values[r][c]}"
    print("\nAll assertions passed. The transformation is working as expected.")

    # New experiment: Test a simple case with clear border and adjacent colors
    print("\nSimple border experiment:")
    simple_input = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 2, 0, 3, 5],
        [5, 0, 1, 0, 5],
        [5, 3, 0, 2, 5],
        [5, 5, 5, 5, 5]
    ])
    print("Input grid:")
    print(visualize_grid(simple_input))
    simple_result = solve_776ffc46(simple_input)
    print("\nResult grid:")
    print(visualize_grid(simple_result))

    # New experiment: Test a simple case with clear border and adjacent colors
    print("\nSimple border experiment:")
    simple_input = ColoredGrid(values=[
        [5, 5, 5, 5, 5],
        [5, 2, 0, 3, 5],
        [5, 0, 1, 0, 5],
        [5, 3, 0, 2, 5],
        [5, 5, 5, 5, 5]
    ])
    print("Input grid:")
    print(visualize_grid(simple_input))
    simple_result = solve_776ffc46(simple_input)
    print("\nResult grid:")
    print(visualize_grid(simple_result))
