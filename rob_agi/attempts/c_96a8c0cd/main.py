from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, Set

def solve_96a8c0cd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal tree-like structure
    of red (2) cells connecting all non-black colored cells. The algorithm starts from
    the top-left corner and moves only right and down, creating paths to connect
    colored cells. Any isolated colored cells are then connected to the nearest part
    of the red network.

    1. Use DFS to create the initial red network.
    2. Connect any remaining unvisited colored cells to the nearest red cell.
    3. Add vertical/horizontal red lines at the right/bottom if needed.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the red network connecting colored cells.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    visited: Set[Tuple[int, int]] = set()

    def is_colored(r: int, c: int) -> bool:
        return grid.get_cell(r, c) > 0

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def connect_right(r: int, c: int) -> None:
        while c + 1 < cols and not is_colored(r, c + 1):
            c += 1
            grid.set_cell(r, c, 2)

    def connect_down(r: int, c: int) -> None:
        while r + 1 < rows and not is_colored(r + 1, c):
            r += 1
            grid.set_cell(r, c, 2)

    def dfs(r: int, c: int) -> None:
        if not is_valid(r, c) or (r, c) in visited:
            return
        
        if is_colored(r, c):
            visited.add((r, c))
            connect_right(r, c)
            connect_down(r, c)
            
            dfs(r, c + 1)  # Move right
            dfs(r + 1, c)  # Move down

    # Start DFS from top-left corner
    dfs(0, 0)

    # Connect any remaining unvisited colored cells
    for r in range(rows):
        for c in range(cols):
            if is_colored(r, c) and (r, c) not in visited:
                # Find nearest red cell and connect
                for dr in range(rows):
                    for dc in range(cols):
                        if is_valid(r - dr, c - dc) and grid.get_cell(r - dr, c - dc) == 2:
                            grid.set_cell(r, c - dc, 2)  # Connect horizontally
                            for rr in range(r - dr, r + 1):
                                grid.set_cell(rr, c - dc, 2)  # Connect vertically
                            visited.add((r, c))
                            break
                    if (r, c) in visited:
                        break

    # Add vertical red line at the right if needed
    if all(grid.get_cell(r, cols - 1) == 0 for r in range(rows)):
        for r in range(rows):
            if grid.get_cell(r, cols - 2) == 2:
                grid.set_cell(r, cols - 1, 2)

    # Add horizontal red line at the bottom if needed
    if all(grid.get_cell(rows - 1, c) == 0 for c in range(cols)):
        for c in range(cols):
            if grid.get_cell(rows - 2, c) == 2:
                grid.set_cell(rows - 1, c, 2)

    return grid
