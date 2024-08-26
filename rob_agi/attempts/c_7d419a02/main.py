from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_7d419a02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue (8) regions to yellow (4) based on their position, structure, and size.
    
    The transformation follows these rules:
    1. Blue regions closer to the edges and corners are more likely to be changed to yellow.
    2. A blue "core" is maintained in the center of the grid, with a cross/plus shape for larger grids.
    3. Line-like structures and small isolated blue regions tend to remain blue.
    4. Black (0) and magenta (6) cells remain unchanged.
    5. The transformation is applied consistently and symmetrically across the entire grid.
    6. Larger blue regions are more likely to be transformed than smaller ones.
    7. The central cross structure is preserved more strongly in larger grids.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    max_distance = math.sqrt(center_row**2 + center_col**2)
    cross_threshold = max(rows, cols) * 0.3
    processed = set()

    def distance_from_center(r: int, c: int) -> float:
        return math.sqrt((r - center_row)**2 + (c - center_col)**2)

    def is_edge_or_corner(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def is_line_like(region: List[Tuple[int, int]]) -> bool:
        if len(region) <= 2:
            return True
        r_min, r_max = min(r for r, _ in region), max(r for r, _ in region)
        c_min, c_max = min(c for _, c in region), max(c for _, c in region)
        return (r_max - r_min <= 1) or (c_max - c_min <= 1)

    def is_part_of_cross(r: int, c: int) -> bool:
        return (abs(r - center_row) <= 1 or abs(c - center_col) <= 1) and distance_from_center(r, c) <= cross_threshold

    def should_transform(region: List[Tuple[int, int]]) -> bool:
        if is_line_like(region) or len(region) == 1:
            return False
        avg_distance = sum(distance_from_center(r, c) for r, c in region) / len(region)
        edge_factor = sum(1 for r, c in region if is_edge_or_corner(r, c)) / len(region)
        size_factor = min(1, len(region) / (rows * cols * 0.05))
        cross_factor = sum(1 for r, c in region if is_part_of_cross(r, c)) / len(region)
        
        transform_score = (avg_distance / max_distance) * 0.4 + edge_factor * 0.3 + size_factor * 0.3 - cross_factor * 0.5
        return transform_score > 0.4

    def flood_fill(row: int, col: int) -> List[Tuple[int, int]]:
        stack = [(row, col)]
        region = []
        while stack:
            r, c = stack.pop()
            if (r, c) not in processed and is_blue(grid.values[r][c]):
                processed.add((r, c))
                region.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return region

    for row in range(rows):
        for col in range(cols):
            if is_blue(grid.values[row][col]) and (row, col) not in processed:
                region = flood_fill(row, col)
                if should_transform(region):
                    for r, c in region:
                        grid.values[r][c] = 4  # Change to yellow

    return grid

def is_blue(cell: int) -> bool:
    return cell == 8
