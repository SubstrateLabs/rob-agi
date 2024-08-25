from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b457fec5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by filling gray areas with a diagonal pattern of colors.
    
    The solution follows these steps:
    1. Identify the color cluster and create an ordered list of colors.
    2. Determine the fill direction based on the color cluster's position.
    3. Find the starting point for filling.
    4. Create a virtual grid and fill it with the diagonal pattern.
    5. Map the virtual grid back to the actual grid, replacing gray cells.
    6. Return the transformed grid.
    
    The pattern starts from the top of the gray region closest to the color cluster,
    uses colors in the order they appear in the input, and follows a diagonal pattern
    while maintaining color sequence across all gray regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Step 1: Identify color cluster and create color list
    color_sequence = [val for row in input_grid.values for val in row if val not in [0, 5]]
    if not color_sequence:
        return output_grid  # No colors to fill with
    
    # Step 2: Determine fill direction
    color_positions = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] in color_sequence]
    avg_col = sum(c for _, c in color_positions) / len(color_positions)
    fill_direction = 1 if avg_col < cols / 2 else -1
    
    # Step 3: Find starting point
    gray_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5]
    if not gray_cells:
        return output_grid  # No gray cells to fill
    start_r = min(r for r, _ in gray_cells)
    start_c = min(c for r, c in gray_cells if r == start_r) if fill_direction == 1 else max(c for r, c in gray_cells if r == start_r)
    
    # Step 4: Create and fill virtual grid
    virtual_rows = max(r for r, _ in gray_cells) - start_r + 1
    virtual_cols = max(abs(c - start_c) for _, c in gray_cells) + 1
    virtual_grid = [[None for _ in range(virtual_cols)] for _ in range(virtual_rows)]
    
    color_index = 0
    for i in range(virtual_rows + virtual_cols - 1):
        for j in range(max(0, i - virtual_rows + 1), min(i + 1, virtual_cols)):
            r, c = i - j, j if fill_direction == 1 else i - j, virtual_cols - 1 - j
            if 0 <= r < virtual_rows and 0 <= c < virtual_cols:
                virtual_grid[r][c] = color_sequence[color_index % len(color_sequence)]
                color_index += 1
    
    # Step 5: Map virtual grid to actual grid
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 5:
                vr, vc = r - start_r, abs(c - start_c)
                if 0 <= vr < virtual_rows and 0 <= vc < virtual_cols:
                    output_grid.values[r][c] = virtual_grid[vr][vc]
    
    return output_grid
