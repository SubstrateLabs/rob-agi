from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f8be4b64(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored centers into territories.
    
    1. Identifies colored centers (non-green cells adjacent to green cells).
    2. Establishes territories based on colored centers, with higher-numbered colors taking priority.
    3. Fills territories by expanding from columns vertically and horizontally.
    4. Restores original green cells.
    5. Removes isolated green cells.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    colored_centers = find_colored_centers(input_grid)
    
    # Sort colored centers by color value in descending order
    colored_centers.sort(key=lambda x: x[2], reverse=True)
    
    # Establish territories
    for r, c, color in colored_centers:
        fill_territory(new_grid, r, c, color)
    
    # Restore green cells
    restore_green_cells(new_grid, input_grid)
    
    # Remove isolated green cells
    remove_isolated_green_cells(new_grid)
    
    return new_grid

def find_colored_centers(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    centers = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0 and grid.values[r][c] != 3:
                if is_adjacent_to_green(grid, r, c):
                    centers.append((r, c, grid.values[r][c]))
    return centers

def is_adjacent_to_green(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 3:
            return True
    return False

def fill_territory(grid: ColoredGrid, r: int, c: int, color: int):
    rows, cols = grid.get_dimensions()
    # Fill the column
    for row in range(rows):
        if grid.values[row][c] == 0 or color > grid.values[row][c]:
            grid.values[row][c] = color
    
    # Expand horizontally
    for row in range(rows):
        # Expand left
        for col in range(c - 1, -1, -1):
            if grid.values[row][col] == 0 or color > grid.values[row][col]:
                grid.values[row][col] = color
            else:
                break
        # Expand right
        for col in range(c + 1, cols):
            if grid.values[row][col] == 0 or color > grid.values[row][col]:
                grid.values[row][col] = color
            else:
                break

def restore_green_cells(new_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 3:
                new_grid.values[r][c] = 3

def remove_isolated_green_cells(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 3:
                if not any(0 <= r + dr < rows and 0 <= c + dc < cols and grid.values[r + dr][c + dc] == 3
                           for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    # Change to the color of any non-green adjacent cell
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] != 0 and grid.values[nr][nc] != 3:
                            grid.values[r][c] = grid.values[nr][nc]
                            break
                    else:
                        # If all adjacent cells are black or out of bounds, leave it green
                        pass
