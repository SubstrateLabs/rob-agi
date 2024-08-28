from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_baf41dbf.main import solve_baf41dbf

def visualize_scaling(input_grid: ColoredGrid, output_grid: ColoredGrid):
    """
    Visualize the scaling of the internal structure from input to output grid.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = output_grid.get_dimensions()

    print("Input Grid:")
    for r in range(input_rows):
        print("".join(str(input_grid.get_cell(r, c)) for c in range(input_cols)))

    print("\nOutput Grid:")
    for r in range(output_rows):
        print("".join(str(output_grid.get_cell(r, c)) for c in range(output_cols)))

# Test with example_0
input_grid = ColoredGrid(values=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
    [0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

output_grid = solve_baf41dbf(input_grid)
visualize_scaling(input_grid, output_grid)
