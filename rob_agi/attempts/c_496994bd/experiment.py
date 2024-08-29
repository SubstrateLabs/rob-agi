from rob_agi.colored_grid import ColoredGrid

def solve_496994bd_experiment(input_grid: ColoredGrid) -> ColoredGrid:
    height, width = input_grid.get_dimensions()
    
    # Count consecutive non-zero rows from the top
    non_zero_rows = 0
    for row in range(height):
        if any(input_grid.get_cell(row, col) != 0 for col in range(width)):
            non_zero_rows += 1
        else:
            break
    
    # Create a deep copy of the input grid
    output = input_grid.deep_copy()
    
    # Mirror the non-zero rows to the bottom
    for i in range(non_zero_rows):
        source_row = i
        target_row = height - non_zero_rows + i
        for col in range(width):
            value = input_grid.get_cell(source_row, col)
            output.set_cell(target_row, col, value)
    
    return output

# Test cases
test_cases = [
    ColoredGrid(values=[
        [2, 2, 2],
        [2, 2, 2],
        [3, 3, 3],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]),
    ColoredGrid(values=[
        [2, 2, 2, 2, 2],
        [8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    ColoredGrid(values=[
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [0, 0, 0],
        [0, 0, 0]
    ]),
    ColoredGrid(values=[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0],
        [0, 0, 0]
    ])
]

expected_outputs = [
    ColoredGrid(values=[
        [2, 2, 2],
        [2, 2, 2],
        [3, 3, 3],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [3, 3, 3],
        [2, 2, 2],
        [2, 2, 2]
    ]),
    ColoredGrid(values=[
        [2, 2, 2, 2, 2],
        [8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8],
        [2, 2, 2, 2, 2]
    ]),
    ColoredGrid(values=[
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [3, 3, 3],
        [4, 4, 4]
    ]),
    ColoredGrid(values=[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 2, 3],
        [4, 5, 6]
    ])
]

for i, (input_grid, expected_output) in enumerate(zip(test_cases, expected_outputs)):
    result = solve_496994bd_experiment(input_grid)
    print(f"Test case {i + 1}:")
    print("Input:")
    print(input_grid)
    print("Expected output:")
    print(expected_output)
    print("Actual output:")
    print(result)
    print("Correct:", result == expected_output)
    print()
