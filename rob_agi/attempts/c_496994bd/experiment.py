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

def visualize_grid(grid: ColoredGrid) -> str:
    height, width = grid.get_dimensions()
    result = ""
    for row in range(height):
        for col in range(width):
            value = grid.get_cell(row, col)
            result += f"{value:2d}"
        result += "\n"
    return result

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
    ]),
    # New test cases
    ColoredGrid(values=[
        [1, 2, 3],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]),
    ColoredGrid(values=[
        [1, 2],
        [3, 4],
        [5, 6],
        [0, 0],
        [0, 0]
    ]),
    ColoredGrid(values=[
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ]),
    ColoredGrid(values=[
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
]

for i, input_grid in enumerate(test_cases):
    result = solve_496994bd_experiment(input_grid)
    print(f"Test case {i + 1}:")
    print("Input:")
    print(visualize_grid(input_grid))
    print("Output:")
    print(visualize_grid(result))
    print()
