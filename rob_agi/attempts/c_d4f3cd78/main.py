from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_d4f3cd78(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the d4f3cd78 challenge by filling the interior of a gray (5) shape with sky blue (8).
    If there's an opening in the shape, it extends the fill through that opening.
    
    The solution follows these steps:
    1. Find the bounding box of the gray shape.
    2. Locate an opening in the shape.
    3. Flood fill the interior with sky blue.
    4. Extend the fill through the opening if it exists.
    
    Args:
        input_grid (ColoredGrid): The input grid containing the shape to be filled.
    
    Returns:
        ColoredGrid: The solved grid with the interior filled and extended if applicable.
    """
    def find_bounding_box(grid: List[List[int]], color: int) -> Optional[Tuple[int, int, int, int]]:
        rows, cols = len(grid), len(grid[0])
        top, left, bottom, right = rows, cols, -1, -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    top, left = min(top, r), min(left, c)
                    bottom, right = max(bottom, r), max(right, c)
        return (top, left, bottom, right) if bottom != -1 else None

    def find_opening(grid: List[List[int]], bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        top, left, bottom, right = bbox
        for r in range(top, bottom + 1):
            if grid[r][right] == 0:
                return r, right
        for c in range(left, right + 1):
            if grid[bottom][c] == 0:
                return bottom, c
        return None

    def flood_fill(grid: List[List[int]], r: int, c: int, new_color: int, bbox: Tuple[int, int, int, int]):
        top, left, bottom, right = bbox
        original_color = grid[r][c]
        if original_color == new_color or original_color == 5:
            return
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if top <= x <= bottom and left <= y <= right and grid[x][y] == original_color:
                grid[x][y] = new_color
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if top <= nx <= bottom and left <= ny <= right and grid[nx][ny] != 5:
                        stack.append((nx, ny))

    def extend_fill(grid: List[List[int]], start_r: int, start_c: int, color: int):
        rows, cols = len(grid), len(grid[0])
        # Extend horizontally
        for c in range(start_c, cols):
            if grid[start_r][c] == 5:
                break
            grid[start_r][c] = color
        # Extend vertically
        for r in range(start_r + 1, rows):
            if grid[r][start_c] == 5:
                break
            grid[r][start_c] = color

    result = input_grid.deep_copy()
    grid = result.values

    bbox = find_bounding_box(grid, 5)
    if not bbox:
        return result

    opening = find_opening(grid, bbox)
    if not opening:
        return result

    # Flood fill from inside the shape
    flood_fill(grid, bbox[0] + 1, bbox[1] + 1, 8, bbox)

    # Extend the fill if there's an opening
    if opening:
        extend_fill(grid, opening[0], opening[1], 8)

    return result
