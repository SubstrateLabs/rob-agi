from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional, Set

def solve_e5c44e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by adding a green 'E' pattern based on the following steps:
    1. Analyze the input grid to find the initial green cell and identify red cells.
    2. Determine the potential 'E' shape boundaries considering grid edges and red cells.
    3. Create the vertical line of the 'E' from the leftmost available column.
    4. Create the top, middle, and bottom horizontal lines of the 'E', avoiding red cells.
    5. Ensure the 'E' touches at least two edges of the grid by extending lines if necessary.
    6. Optimize the 'E' shape by expanding it where possible, ensuring connectivity.
    7. Clean up the 'E' shape by removing unnecessary green cells and isolated parts.
    8. Verify that all red cells are preserved, the 'E' is connected, and touches at least two edges.

    The function adapts to various initial conditions and red cell placements
    to create the largest possible 'E' pattern that satisfies all challenge requirements,
    including asymmetrical or irregular 'E' shapes when necessary.
    """
    output_grid = input_grid.deep_copy()
    initial_green = find_initial_green(output_grid)
    if not initial_green:
        return output_grid

    left_column = find_leftmost_column(output_grid)
    create_vertical_line(output_grid, left_column)
    create_horizontal_lines(output_grid, left_column, initial_green)
    ensure_two_edge_contact(output_grid, left_column)
    optimize_e_shape(output_grid, left_column, initial_green)
    clean_up_e_shape(output_grid)
    connect_disconnected_parts(output_grid)

    # Verify the solution
    if not verify_solution(output_grid, input_grid):
        # If verification fails, revert to the input grid
        return input_grid

    return output_grid

def verify_solution(output_grid: ColoredGrid, input_grid: ColoredGrid) -> bool:
    # Check if all red cells are preserved
    for r in range(output_grid.num_rows):
        for c in range(output_grid.num_cols):
            if input_grid.get_cell(r, c) == 2 and output_grid.get_cell(r, c) != 2:
                return False

    # Check if the 'E' is connected and touches at least two edges
    green_cells = [(r, c) for r in range(output_grid.num_rows) for c in range(output_grid.num_cols) if output_grid.get_cell(r, c) == 3]
    if not green_cells:
        return False

    connected_cells = flood_fill(output_grid, green_cells[0][0], green_cells[0][1])
    if len(connected_cells) != len(green_cells):
        return False

    edges_touched = sum([
        any(output_grid.get_cell(0, c) == 3 for c in range(output_grid.num_cols)),
        any(output_grid.get_cell(output_grid.num_rows-1, c) == 3 for c in range(output_grid.num_cols)),
        any(output_grid.get_cell(r, 0) == 3 for r in range(output_grid.num_rows)),
        any(output_grid.get_cell(r, output_grid.num_cols-1) == 3 for r in range(output_grid.num_rows))
    ])
    return edges_touched >= 2

def find_initial_green(grid: ColoredGrid) -> Tuple[int, int]:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                return r, c
    # If no green cell is found, return the center of the grid
    return grid.num_rows // 2, grid.num_cols // 2

def find_leftmost_column(grid: ColoredGrid) -> int:
    for c in range(grid.num_cols):
        if all(grid.get_cell(r, c) != 2 for r in range(grid.num_rows)):
            return c
    return 0  # If no column is free of red cells, start from the leftmost column

def create_vertical_line(grid: ColoredGrid, col: int):
    for r in range(grid.num_rows):
        if grid.get_cell(r, col) == 0:
            grid.set_cell(r, col, 3)

def create_horizontal_lines(grid: ColoredGrid, left_col: int, initial_green: Tuple[int, int]):
    rows, cols = grid.num_rows, grid.num_cols
    initial_row, _ = initial_green

    # Top line
    top_row = min(r for r in range(rows) if grid.get_cell(r, left_col) != 2)
    for c in range(left_col + 1, cols):
        if grid.get_cell(top_row, c) == 2:
            break
        grid.set_cell(top_row, c, 3)

    # Middle line
    for c in range(left_col + 1, cols):
        if grid.get_cell(initial_row, c) == 2:
            break
        grid.set_cell(initial_row, c, 3)

    # Bottom line
    bottom_row = max(r for r in range(rows) if grid.get_cell(r, left_col) != 2)
    for c in range(left_col + 1, cols):
        if grid.get_cell(bottom_row, c) == 2:
            break
        grid.set_cell(bottom_row, c, 3)

    # Fill in the vertical line
    for r in range(top_row, bottom_row + 1):
        if grid.get_cell(r, left_col) == 0:
            grid.set_cell(r, left_col, 3)

def ensure_two_edge_contact(grid: ColoredGrid, left_col: int):
    rows, cols = grid.num_rows, grid.num_cols
    edges_touched = sum([
        any(grid.get_cell(0, c) == 3 for c in range(cols)),
        any(grid.get_cell(rows-1, c) == 3 for c in range(cols)),
        any(grid.get_cell(r, 0) == 3 for r in range(rows)),
        any(grid.get_cell(r, cols-1) == 3 for r in range(rows))
    ])
    
    if edges_touched < 2:
        # Extend top line to right edge
        top_row = min(r for r in range(rows) if grid.get_cell(r, left_col) == 3)
        for c in range(cols-1, left_col, -1):
            if grid.get_cell(top_row, c) == 0:
                grid.set_cell(top_row, c, 3)
        
        # Extend bottom horizontal line to right edge
        bottom_row = max(r for r in range(rows) if grid.get_cell(r, left_col) == 3)
        for c in range(cols-1, left_col, -1):
            if grid.get_cell(bottom_row, c) == 0:
                grid.set_cell(bottom_row, c, 3)

def connect_disconnected_parts(grid: ColoredGrid):
    for c in range(1, grid.num_cols - 1):
        connected = False
        for r in range(grid.num_rows):
            if grid.get_cell(r, c) == 3:
                if not connected:
                    connected = True
                elif grid.get_cell(r - 1, c) != 3:
                    for i in range(r - 1, -1, -1):
                        if grid.get_cell(i, c) == 3:
                            break
                        if grid.get_cell(i, c) == 0:
                            grid.set_cell(i, c, 3)

def optimize_e_shape(grid: ColoredGrid, left_col: int, initial_green: Tuple[int, int]):
    rows, cols = grid.num_rows, grid.num_cols
    initial_row, initial_col = initial_green

    # Determine the extent of the vertical line
    top_row = min(r for r in range(rows) if grid.get_cell(r, left_col) == 3)
    bottom_row = max(r for r in range(rows) if grid.get_cell(r, left_col) == 3)

    # Optimize top horizontal line
    for c in range(left_col + 1, cols):
        if grid.get_cell(top_row, c) == 2:
            break
        grid.set_cell(top_row, c, 3)

    # Optimize middle horizontal line
    for c in range(left_col + 1, cols):
        if grid.get_cell(initial_row, c) == 2:
            break
        grid.set_cell(initial_row, c, 3)

    # Optimize bottom horizontal line
    for c in range(left_col + 1, cols):
        if grid.get_cell(bottom_row, c) == 2:
            break
        grid.set_cell(bottom_row, c, 3)

    # Try to extend vertical line to the right
    for c in range(left_col + 1, cols):
        if all(grid.get_cell(r, c) in [0, 3] for r in range(top_row, bottom_row + 1)):
            for r in range(top_row, bottom_row + 1):
                if grid.get_cell(r, c-1) == 3 and grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 3)
        else:
            break

def clean_up_e_shape(grid: ColoredGrid):
    rows, cols = grid.num_rows, grid.num_cols
    
    # Remove unnecessary green cells
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                neighbors = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) == 3)
                if neighbors <= 1:
                    grid.set_cell(r, c, 0)
    
    # Ensure clear spaces inside the 'E'
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (grid.get_cell(r-1, c) == 3 and grid.get_cell(r+1, c) == 3 and
                grid.get_cell(r, c-1) == 3 and grid.get_cell(r, c+1) == 3):
                grid.set_cell(r, c, 0)
    
    # Remove any disconnected green cells
    connected = set()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                if not connected:
                    connected = flood_fill(grid, r, c)
                elif (r, c) not in connected:
                    grid.set_cell(r, c, 0)

def flood_fill(grid: ColoredGrid, r: int, c: int) -> Set[Tuple[int, int]]:
    rows, cols = grid.num_rows, grid.num_cols
    connected = set()
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        if (r, c) not in connected and grid.get_cell(r, c) == 3:
            connected.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    stack.append((nr, nc))
    return connected
