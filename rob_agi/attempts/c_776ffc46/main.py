from rob_agi.colored_grid import ColoredGrid


def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions with more than one cell change to Red (2)
    2. Red (2) regions with more than one cell remain Red (2)
    3. Green (3) regions with more than one cell remain Green (3)
    4. Single-cell regions and other colors remain unchanged

    Approach:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Iterate through the grid to find connected regions for Blue (1), Red (2), and Green (3) colors.
    3. For each region:
       a. If it's Blue (1) and has more than one cell, change it to Red (2).
       b. If it's Red (2) or Green (3) and has more than one cell, keep it unchanged.
       c. For single-cell regions or other colors, do nothing.
    4. Return the transformed grid.

    This implementation ensures that all Blue regions are processed independently,
    addressing the issue where multiple Blue regions needed to be transformed.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()

    def dfs(r, c, color):
        region = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and output_grid.values[curr_r][curr_c] == color:
                visited.add((curr_r, curr_c))
                region.append((curr_r, curr_c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                color = output_grid.values[r][c]
                if color in [1, 2, 3]:  # Blue, Red, Green
                    region = dfs(r, c, color)
                    if len(region) > 1:
                        new_color = 2 if color == 1 else color  # Change Blue to Red, keep others
                        for cell_r, cell_c in region:
                            output_grid.set_cell(cell_r, cell_c, new_color)

    return output_grid
