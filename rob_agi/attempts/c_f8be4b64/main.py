from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_f8be4b64(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored crosses into territories.
    
    1. Identifies colored crosses in the input grid.
    2. Determines color priority (descending order, excluding green).
    3. Calculates territories for each cross by column.
    4. Fills a new grid based on territories and color priority.
    5. Restores original green crosses.
    6. Removes isolated green cells.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    crosses = find_crosses(input_grid)
    color_priority = get_color_priority(crosses)
    territories = calculate_territories(crosses, rows, cols)
    
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    for color in color_priority:
        for cross in crosses:
            if cross[2] == color:
                fill_territory(new_grid, territories[cross[:2]], color, rows)
    
    restore_green_crosses(new_grid, crosses)
    remove_isolated_green_cells(new_grid)
    
    return new_grid

def find_crosses(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    crosses = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0 and is_cross_center(grid, r, c):
                crosses.append((r, c, grid.values[r][c]))
    return crosses

def is_cross_center(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 3:
            return True
    return False

def get_color_priority(crosses: List[Tuple[int, int, int]]) -> List[int]:
    colors = set(cross[2] for cross in crosses if cross[2] != 3)
    return sorted(list(colors), reverse=True)

def calculate_territories(crosses: List[Tuple[int, int, int]], rows: int, cols: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    territories = {}
    for c in range(cols):
        column_crosses = sorted([cross for cross in crosses if cross[1] == c], key=lambda x: x[0])
        for i, (r, _, _) in enumerate(column_crosses):
            top = 0 if i == 0 else (r + column_crosses[i-1][0]) // 2
            bottom = rows - 1 if i == len(column_crosses) - 1 else (r + column_crosses[i+1][0]) // 2
            territories[(r, c)] = (top, bottom)
    return territories

def fill_territory(grid: ColoredGrid, territory: Tuple[int, int], color: int, rows: int):
    top, bottom = territory
    c = territory[1]
    for r in range(rows):
        if grid.values[r][c] == 0 or color > grid.values[r][c]:
            grid.values[r][c] = color
    for r in range(top, bottom + 1):
        for c in range(len(grid.values[0])):
            if grid.values[r][c] == 0 or color > grid.values[r][c]:
                grid.values[r][c] = color

def restore_green_crosses(grid: ColoredGrid, crosses: List[Tuple[int, int, int]]):
    for r, c, color in crosses:
        grid.values[r][c] = color
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                grid.values[nr][nc] = 3

def remove_isolated_green_cells(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 3:
                if not any(0 <= r + dr < rows and 0 <= c + dc < cols and grid.values[r + dr][c + dc] == 3
                           for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    grid.values[r][c] = 0
