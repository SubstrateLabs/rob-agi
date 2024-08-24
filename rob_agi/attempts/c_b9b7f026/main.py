from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple, Set

def solve_b9b7f026(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by finding the shape with exactly one hole and the smallest area.
    
    The function performs the following steps:
    1. Identifies all shapes in the grid using flood fill.
    2. Counts the number of holes in each shape.
    3. Selects shapes with exactly one hole.
    4. Among these, chooses the shape with the smallest area.
    5. Returns a 1x1 grid with the color of the chosen shape.
    
    If no shape with one hole is found, it returns a 1x1 grid with color 0 (black).
    """
    def flood_fill(grid: List[List[int]], start: Tuple[int, int], color: int) -> Set[Tuple[int, int]]:
        rows, cols = len(grid), len(grid[0])
        visited = set()
        queue = deque([start])
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == color and (nr, nc) not in visited:
                    queue.append((nr, nc))
        return visited

    def count_holes(grid: List[List[int]], region: Set[Tuple[int, int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        visited = set(region)
        holes = 0

        def is_enclosed(hole: Set[Tuple[int, int]]) -> bool:
            for r, c in hole:
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    return False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in hole and (nr, nc) not in region:
                        return False
            return True

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r][c] == 0:
                    hole = flood_fill(grid, (r, c), 0)
                    if is_enclosed(hole):
                        holes += 1
                    visited |= hole

        return holes

    def find_shapes_with_holes(grid: List[List[int]]) -> List[Tuple[int, int, int]]:
        rows, cols = len(grid), len(grid[0])
        visited = set()
        shapes = []

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 0 and (r, c) not in visited:
                    color = grid[r][c]
                    region = flood_fill(grid, (r, c), color)
                    visited |= region
                    holes = count_holes(grid, region)
                    shapes.append((color, len(region), holes))

        return shapes

    grid = input_grid.values
    shapes = find_shapes_with_holes(grid)
    
    one_hole_shapes = [(color, area) for color, area, holes in shapes if holes == 1]
    
    if not one_hole_shapes:
        return ColoredGrid(values=[[0]])

    min_area = min(area for _, area in one_hole_shapes)
    result_color = min(color for color, area in one_hole_shapes if area == min_area)
    return ColoredGrid(values=[[result_color]])
