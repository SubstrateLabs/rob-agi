from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_7d419a02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue (8) regions to yellow (4) based on their position, structure, and size.
    
    The transformation follows these rules:
    1. Identifies structural elements like central cross and checkerboard patterns.
    2. Segments the grid into regions and analyzes their context.
    3. Assigns transformation scores based on region size, location, and grid structure.
    4. Transforms regions to yellow starting from highest scores until a target ratio is reached.
    5. Refines the pattern to improve consistency and preserve small features.
    6. Ensures symmetry across vertical and horizontal axes.
    7. Maintains a balance between blue and yellow based on grid size.
    8. Preserves black (0) and magenta (6) cells.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    max_distance = math.sqrt(center_row**2 + center_col**2)
    scale_factor = math.log(max(rows, cols)) / math.log(30)  # Normalized to 1 for 30x30 grid
    
    def distance_from_center(r: int, c: int) -> float:
        return math.sqrt((r - center_row)**2 + (c - center_col)**2) / max_distance

    def is_part_of_structure(r: int, c: int) -> bool:
        cross_width = max(1, min(rows, cols) // 10)
        is_cross = abs(r - center_row) <= cross_width or abs(c - center_col) <= cross_width
        is_checkerboard = (r + c) % 2 == 0
        return is_cross or is_checkerboard

    def flood_fill(row: int, col: int) -> List[Tuple[int, int]]:
        stack, region = [(row, col)], []
        while stack:
            r, c = stack.pop()
            if (r, c) not in region and grid.values[r][c] == 8:
                region.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return region

    def get_context_score(region: List[Tuple[int, int]]) -> float:
        context_cells = set((r+dr, c+dc) for r, c in region for dr in [-1, 0, 1] for dc in [-1, 0, 1])
        context_cells -= set(region)
        yellow_count = sum(1 for r, c in context_cells if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 4)
        return yellow_count / len(context_cells) if context_cells else 0

    def calculate_transform_score(region: List[Tuple[int, int]]) -> float:
        avg_distance = sum(distance_from_center(r, c) for r, c in region) / len(region)
        size_factor = min(1, len(region) / (rows * cols * 0.05))
        structure_factor = sum(1 for r, c in region if is_part_of_structure(r, c)) / len(region)
        return (avg_distance * 0.4 + size_factor * 0.4 - structure_factor * 0.8) * scale_factor

    regions = []
    for row in range(rows):
        for col in range(cols):
            if grid.values[row][col] == 8 and not any((row, col) in region for region in regions):
                regions.append(flood_fill(row, col))

    regions.sort(key=lambda r: calculate_transform_score(r) + get_context_score(r), reverse=True)
    target_yellow_ratio = 0.4 + 0.2 * scale_factor
    current_yellow_ratio = 0

    for region in regions:
        if current_yellow_ratio >= target_yellow_ratio:
            break
        symmetric_cells = set((r, c) for r, c in region).union(
            (rows-1-r, c) for r, c in region).union(
            (r, cols-1-c) for r, c in region).union(
            (rows-1-r, cols-1-c) for r, c in region)
        )
        for r, c in symmetric_cells:
            if grid.values[r][c] == 8:
                grid.values[r][c] = 4
                current_yellow_ratio += 1 / (rows * cols)

    # Pattern refinement
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:
                neighbors = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                yellow_neighbors = sum(1 for nr, nc in neighbors if grid.values[nr][nc] == 4)
                if yellow_neighbors >= 3:
                    grid.values[r][c] = 4

    return grid
