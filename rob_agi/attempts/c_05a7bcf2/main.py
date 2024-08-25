from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_05a7bcf2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. Expands all non-sky blue colors in all directions until hitting a sky blue barrier, another color, or the edge.
    2. Fills remaining empty cells with sky blue.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid_cell(row: int, col: int) -> bool:
        return 0 <= row < rows and 0 <= col < cols

    def get_adjacent_cells(row: int, col: int) -> List[Tuple[int, int]]:
        return [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]

    def bfs_color_expansion(start_row: int, start_col: int, color: int):
        queue = [(start_row, start_col)]
        visited = set()
        
        while queue:
            row, col = queue.pop(0)
            if (row, col) in visited:
                continue
            
            visited.add((row, col))
            
            for adj_row, adj_col in get_adjacent_cells(row, col):
                if is_valid_cell(adj_row, adj_col) and grid.values[adj_row][adj_col] in [0, color]:
                    grid.values[adj_row][adj_col] = color
                    queue.append((adj_row, adj_col))

    for row in range(rows):
        for col in range(cols):
            if grid.values[row][col] not in [0, 8]:  # If cell is a color other than black or sky blue
                bfs_color_expansion(row, col, grid.values[row][col])

    # Fill remaining black cells with sky blue
    for row in range(rows):
        for col in range(cols):
            if grid.values[row][col] == 0:
                grid.values[row][col] = 8

    return grid
