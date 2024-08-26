from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c62e2108(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the c62e2108 challenge by expanding patterns across the grid.
    
    The function identifies non-black, non-blue shapes in the input grid,
    expands them based on their position in the grid:
    - Top-left shapes are expanded to a 2x3 grid with a connecting top line.
    - Bottom-left shapes are expanded vertically to form a column of 4 repetitions.
    - Top-right shapes are expanded down to form a 2x1 column.
    - Bottom-right shapes are expanded to a 2x3 grid with an extra right column and connecting top line.
    Blue areas are removed, and black areas outside expansions are preserved.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with expanded patterns.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    shapes = find_shapes(input_grid)
    
    for shape in shapes:
        expand_shape(new_grid, shape, rows, cols)
    
    return ColoredGrid(values=new_grid)

def find_shapes(grid: ColoredGrid) -> List[Dict]:
    shapes = []
    rows, cols = grid.get_dimensions()
    visited = set()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] > 1:  # Non-black, non-blue
                shape = get_shape(grid, r, c, visited)
                shapes.append(shape)
    
    return sorted(shapes, key=lambda x: (x['row'], x['col']), reverse=True)

def get_shape(grid: ColoredGrid, start_r: int, start_c: int, visited: set) -> Dict:
    color = grid.values[start_r][start_c]
    shape = {'color': color, 'row': start_r, 'col': start_c, 'pixels': []}
    stack = [(start_r, start_c)]
    
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and grid.values[r][c] == color:
            visited.add((r, c))
            shape['pixels'].append((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                    stack.append((nr, nc))
    
    return shape

def expand_shape(grid: List[List[int]], shape: Dict, rows: int, cols: int):
    r, c = shape['row'], shape['col']
    color = shape['color']
    mid_row, mid_col = rows // 2, cols // 2
    
    if r < mid_row and c < mid_col:  # Top-left quadrant
        expand_top_left(grid, color, r, c, rows, cols)
    elif r >= mid_row and c < mid_col:  # Bottom-left quadrant
        expand_bottom_left(grid, color, r, c, rows, cols)
    elif r < mid_row and c >= mid_col:  # Top-right quadrant
        expand_top_right(grid, color, r, c, rows, cols)
    else:  # Bottom-right quadrant
        expand_bottom_right(grid, color, r, c, rows, cols)

def expand_top_left(grid: List[List[int]], color: int, r: int, c: int, rows: int, cols: int):
    for i in range(r, min(r + 6, rows)):
        for j in range(c, min(c + 15, cols)):
            if i == r or (i - r) % 4 == 0 or (j - c) % 4 == 0 or (j - c) % 4 == 3:
                grid[i][j] = color

def expand_bottom_left(grid: List[List[int]], color: int, r: int, c: int, rows: int, cols: int):
    for i in range(max(0, r - 12), rows):
        for j in range(c, min(c + 4, cols)):
            if (i - r) % 4 == 0 or (i - r) % 4 == 3 or j == c or j == c + 3:
                grid[i][j] = color

def expand_top_right(grid: List[List[int]], color: int, r: int, c: int, rows: int, cols: int):
    for i in range(r, min(r + 8, rows)):
        for j in range(c, min(c + 4, cols)):
            if (i - r) % 4 == 0 or (i - r) % 4 == 3 or j == c or j == c + 3:
                grid[i][j] = color

def expand_bottom_right(grid: List[List[int]], color: int, r: int, c: int, rows: int, cols: int):
    for i in range(max(0, r - 4), rows):
        for j in range(max(0, c - 12), cols):
            if i == r - 4 or (i - r) % 4 == 0 or (j - c) % 4 == 0 or j == cols - 1:
                grid[i][j] = color
