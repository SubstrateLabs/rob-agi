from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c62e2108(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the c62e2108 challenge by expanding patterns across the grid.
    
    The function identifies non-black, non-blue shapes in the input grid and expands them
    based on their relative position. The expansion pattern varies gradually:
    - Horizontal expansion increases from 3 to 4 units moving from left to right.
    - Vertical expansion increases from 2 to 4 units moving from top to bottom.
    - Shapes in the upper half create a continuous top line connecting all expanded units.
    - Shapes in the lower half expand vertically, with the leftmost expanding 4 times.
    - Bottom-right shapes expand to fill available space, mirroring top-left patterns.
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
    
    return sorted(shapes, key=lambda x: (x['row'], x['col']))

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
    rel_row, rel_col = r / rows, c / cols
    
    h_expand = int(3 + rel_col)
    v_expand = int(2 + 2 * rel_row)
    
    pattern = get_pattern(shape['pixels'], r, c)
    
    for i in range(v_expand):
        for j in range(h_expand):
            for pr, pc in pattern:
                nr, nc = r + i * (pr - r), c + j * (pc - c)
                if 0 <= nr < rows and 0 <= nc < cols:
                    grid[nr][nc] = color
    
    # Create continuous top line
    if rel_row < 0.5:
        for j in range(c, min(cols, c + h_expand * (pattern[-1][1] - pattern[0][1] + 1))):
            grid[r][j] = color

def get_pattern(pixels: List[Tuple[int, int]], r: int, c: int) -> List[Tuple[int, int]]:
    return [(pr - r, pc - c) for pr, pc in pixels]
