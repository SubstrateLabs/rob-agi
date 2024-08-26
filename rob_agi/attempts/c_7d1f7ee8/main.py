from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple

def solve_7d1f7ee8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling or changing colors of enclosed regions.
    
    The function works as follows:
    1. Identifies distinct color regions in the grid.
    2. Processes regions from outermost to innermost.
    3. For each region:
       - If it's a frame, fills its interior with its own color.
       - If it's a solid shape, preserves it.
       - If enclosed by another color, changes to that color if it's not a solid shape.
    4. Recursively processes sub-regions within each region.
    5. Preserves unenclosed black (0) areas and shapes touching the grid border.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    processed = [[False for _ in range(cols)] for _ in range(rows)]
    
    def get_region(r: int, c: int, color: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        boundary = set()
        interior = set()
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if processed[x][y] or output_grid.values[x][y] != color:
                continue
            processed[x][y] = True
            is_boundary = False
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if output_grid.values[nx][ny] != color:
                        is_boundary = True
                    elif not processed[nx][ny]:
                        stack.append((nx, ny))
                else:
                    is_boundary = True
            if is_boundary:
                boundary.add((x, y))
            else:
                interior.add((x, y))
        return boundary, interior
    
    def is_enclosed(boundary: Set[Tuple[int, int]], color: int) -> Tuple[bool, int]:
        enclosing_color = None
        for x, y in boundary:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and output_grid.values[nx][ny] != color:
                    if enclosing_color is None:
                        enclosing_color = output_grid.values[nx][ny]
                    elif output_grid.values[nx][ny] != enclosing_color:
                        return False, 0
        return enclosing_color is not None, enclosing_color
    
    def is_solid_shape(boundary: Set[Tuple[int, int]], interior: Set[Tuple[int, int]]) -> bool:
        min_x = min(x for x, _ in boundary | interior)
        max_x = max(x for x, _ in boundary | interior)
        min_y = min(y for _, y in boundary | interior)
        max_y = max(y for _, y in boundary | interior)
        return len(boundary | interior) == (max_x - min_x + 1) * (max_y - min_y + 1)
    
    def process_region(boundary: Set[Tuple[int, int]], interior: Set[Tuple[int, int]], color: int):
        enclosed, enclosing_color = is_enclosed(boundary, color)
        is_solid = is_solid_shape(boundary, interior)
        
        if enclosed and not is_solid:
            for x, y in boundary | interior:
                output_grid.values[x][y] = enclosing_color
        else:
            for x, y in interior:
                output_grid.values[x][y] = color
        
        # Process sub-regions
        for x, y in interior:
            if not processed[x][y]:
                sub_boundary, sub_interior = get_region(x, y, output_grid.values[x][y])
                process_region(sub_boundary, sub_interior, output_grid.values[x][y])
    
    # Process regions from outermost to innermost
    for r in range(rows):
        for c in range(cols):
            if not processed[r][c] and output_grid.values[r][c] != 0:
                boundary, interior = get_region(r, c, output_grid.values[r][c])
                process_region(boundary, interior, output_grid.values[r][c])
    
    return output_grid
