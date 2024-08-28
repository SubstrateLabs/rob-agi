from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_833dafe3.test import test_833dafe3_example_0, test_833dafe3_example_1

def extract_frame(grid: ColoredGrid) -> ColoredGrid:
    height, width = grid.get_dimensions()
    frame = ColoredGrid(values=[
        [grid.values[i][j] for j in range(width)]
        for i in [0, 1, height-2, height-1]
    ])
    for i in range(2, height-2):
        frame.values.append([grid.values[i][0], grid.values[i][1], grid.values[i][-2], grid.values[i][-1]])
    return frame

def compare_input_output_frame(input_grid: ColoredGrid, expected_output: ColoredGrid):
    print("Input Grid:")
    print(input_grid)
    print("\nExpected Output Grid:")
    print(expected_output)
    print("\nExtracted Frame from Expected Output:")
    frame = extract_frame(expected_output)
    print(frame)
    print("\nFrame Dimensions:", frame.get_dimensions())

print("Example 0:")
test_case = test_833dafe3_example_0
compare_input_output_frame(test_case.input_grid, test_case.expected)

print("\n" + "="*50 + "\n")

print("Example 1:")
test_case = test_833dafe3_example_1
compare_input_output_frame(test_case.input_grid, test_case.expected)
