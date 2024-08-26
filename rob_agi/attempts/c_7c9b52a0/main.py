from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_7c9b52a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting distinct regions and rearranging them in a compact form.

    1. Identifies the background color (most common color on the border)
    2. Extracts distinct regions of non-background colors using flood fill
    3. Sorts regions by color and original position
    4. Creates a new compact grid by placing regions in sorted order, maintaining relative positions
    5. Optimizes the compact grid by removing background-only rows and columns
    6. Returns the new compact grid

    The function preserves the shapes of regions, maintains color order, attempts to keep
    relative positioning of regions, aims for maximum compactness, and works consistently
    regardless of the background color or the colors of the regions.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: A new compact grid containing rearranged non-background regions
    """
    def find_background_color(grid: ColoredGrid) -> int:
        rows, cols = grid.get_dimensions()
        border = (
            grid.values[0] + grid.values[-1] +
            [row[0] for row in grid.values[1:-1]] +
            [row[-1] for row in grid.values[1:-1]]
        )
        return max(set(border), key=border.count)

    def extract_regions(grid: ColoredGrid, bg_color: int) -> List[Tuple[int, List[Tuple[int, int]], Tuple[int, int]]]:
        rows, cols = grid.get_dimensions()
        visited = set()
        regions = []

        def bfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
            queue = deque([(r, c)])
            region = []
            while queue:
                curr_r, curr_c = queue.popleft()
                if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == color:
                    visited.add((curr_r, curr_c))
                    region.append((curr_r - r, curr_c - c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            queue.append((nr, nc))
            return region

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.values[r][c] != bg_color:
                    color = grid.values[r][c]
                    region = bfs(r, c, color)
                    regions.append((color, region, (r, c)))

        return regions

    def create_compact_grid(regions: List[Tuple[int, List[Tuple[int, int]], Tuple[int, int]]], bg_color: int) -> List[List[int]]:
        if not regions:
            return [[]]

        sorted_regions = sorted(regions, key=lambda x: (x[0], x[2]))
        compact_grid = [[bg_color]]

        for color, shape, original_pos in sorted_regions:
            placed = False
            while not placed:
                best_position = find_best_position(compact_grid, shape, original_pos, bg_color)
                if best_position:
                    r, c = best_position
                    place_region(compact_grid, r, c, shape, color)
                    placed = True
                else:
                    compact_grid = expand_grid(compact_grid, bg_color)

        return compact_grid

    def find_best_position(grid: List[List[int]], shape: List[Tuple[int, int]], original_pos: Tuple[int, int], bg_color: int) -> Optional[Tuple[int, int]]:
        rows, cols = len(grid), len(grid[0])
        best_pos = None
        min_distance = float('inf')
        
        for r in range(rows):
            for c in range(cols):
                if can_place_region(grid, r, c, shape, bg_color):
                    distance = abs(r - original_pos[0]) + abs(c - original_pos[1])
                    if distance < min_distance:
                        min_distance = distance
                        best_pos = (r, c)
        
        return best_pos

    def can_place_region(grid: List[List[int]], r: int, c: int, shape: List[Tuple[int, int]], bg_color: int) -> bool:
        rows, cols = len(grid), len(grid[0])
        for dr, dc in shape:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or grid[nr][nc] != bg_color:
                return False
        return True

    def place_region(grid: List[List[int]], r: int, c: int, shape: List[Tuple[int, int]], color: int) -> None:
        for dr, dc in shape:
            grid[r + dr][c + dc] = color

    def expand_grid(grid: List[List[int]], bg_color: int) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0])
        new_rows = rows + 1
        new_cols = cols + 1
        new_grid = [[bg_color for _ in range(new_cols)] for _ in range(new_rows)]
        for r in range(rows):
            for c in range(cols):
                new_grid[r][c] = grid[r][c]
        return new_grid

    def remove_background_rows_and_columns(grid: List[List[int]], bg_color: int) -> List[List[int]]:
        grid = [row for row in grid if any(cell != bg_color for cell in row)]
        cols_to_keep = [col for col in range(len(grid[0])) if any(row[col] != bg_color for row in grid)]
        grid = [[row[col] for col in cols_to_keep] for row in grid]
        return grid

    bg_color = find_background_color(input_grid)
    regions = extract_regions(input_grid, bg_color)
    compact_grid = create_compact_grid(regions, bg_color)
    final_grid = remove_background_rows_and_columns(compact_grid, bg_color)

    return ColoredGrid(values=final_grid)
