from rob_agi.colored_grid import ColoredGrid


def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions with more than one cell change to Red (2)
    2. Red (2) regions remain unchanged
    3. Green (3) regions remain unchanged
    4. All other colors remain unchanged
    5. Single-cell regions of any color remain unchanged

    Approach:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Iterate through the grid to find connected regions of Blue (1) color.
    3. For each Blue region:
       a. If it has more than one cell, change all cells in the region to Red (2).
       b. If it's a single cell, leave it unchanged.
    4. Return the transformed grid.

    This implementation focuses on transforming only the Blue regions with more than one cell,
    while leaving all other colors and single-cell Blue regions unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()

    def dfs(r, c):
        region = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and output_grid.values[curr_r][curr_c] == 1:  # Blue
                visited.add((curr_r, curr_c))
                region.append((curr_r, curr_c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and output_grid.values[r][c] == 1:  # Blue
                region = dfs(r, c)
                if len(region) > 1:
                    for cell_r, cell_c in region:
                        output_grid.set_cell(cell_r, cell_c, 2)  # Change to Red

    return output_grid
