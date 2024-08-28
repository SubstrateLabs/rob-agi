from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def create_sample_grid(red_cells: List[Tuple[int, int]]) -> ColoredGrid:
    grid = ColoredGrid(values=[[0] * 11 for _ in range(11)])
    grid.set_cell(5, 5, 3)  # Set the initial green cell
    for r, c in red_cells:
        grid.set_cell(r, c, 2)
    return grid

def print_grid(grid: ColoredGrid):
    for row in range(grid.num_rows):
        print(" ".join(str(grid.get_cell(row, col)) for col in range(grid.num_cols)))

def create_e_shape(grid: ColoredGrid, red_cells: List[Tuple[int, int]]) -> ColoredGrid:
    rows, cols = grid.num_rows, grid.num_cols
    
    # Find leftmost available column
    left_col = next(c for c in range(cols) if all((r, c) not in red_cells for r in range(rows)))
    
    # Create vertical line
    for r in range(rows):
        if (r, left_col) not in red_cells:
            grid.set_cell(r, left_col, 3)
    
    # Create horizontal lines
    for r in [0, 5, 10]:  # Top, middle, bottom
        for c in range(left_col + 1, cols):
            if (r, c) not in red_cells:
                grid.set_cell(r, c, 3)
            else:
                break
    
    return grid

# Example 1: No red cells
print("Example 1: No red cells")
grid1 = create_sample_grid([])
e_shape1 = create_e_shape(grid1, [])
print_grid(e_shape1)
print()

# Example 2: Some red cells
print("Example 2: Some red cells")
red_cells2 = [(0, 2), (2, 10), (3, 1), (3, 8), (8, 2), (8, 10), (9, 0), (10, 5)]
grid2 = create_sample_grid(red_cells2)
e_shape2 = create_e_shape(grid2, red_cells2)
print_grid(e_shape2)
