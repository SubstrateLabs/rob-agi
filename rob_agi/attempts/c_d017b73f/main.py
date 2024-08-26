from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d017b73f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by compressing it horizontally while preserving color groups.
    
    The function identifies non-black color groups, moves them to the left side of the grid,
    and removes unnecessary black columns. This maintains the vertical structure and alignment
    of color groups while producing the narrowest possible output grid. Color groups are kept
    in their original vertical positions and horizontal order.
    """
    rows, cols = input_grid.get_dimensions()
    color_groups = identify_color_groups(input_grid)
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    current_col = 0

    for group in color_groups:
        group_width = max(col for _, col in group) - min(col for _, col in group) + 1
        for row, col in group:
            new_col = current_col + (col - min(col for _, col in group))
            new_grid[row][new_col] = input_grid.values[row][col]
        current_col += group_width

    # Trim trailing black columns
    while all(row[-1] == 0 for row in new_grid):
        for row in new_grid:
            row.pop()

    return ColoredGrid(values=new_grid)

def identify_color_groups(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    groups = []

    def dfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        group = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == color:
                visited.add((curr_r, curr_c))
                group.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return group

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                group = dfs(r, c, grid.values[r][c])
                groups.append(group)

    return sorted(groups, key=lambda g: min(col for _, col in g))
