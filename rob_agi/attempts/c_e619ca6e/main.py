from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e619ca6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid expansion challenge by identifying green cells in the input grid
    and applying a complex expansion pattern.
    
    The solution follows these steps:
    1. Identify all green (value 3) cells in the input grid as seed points.
    2. Create a new output grid with the same dimensions as the input.
    3. For each seed point, apply an expansion pattern that creates 3x3 green squares
       in a grid-like structure, expanding horizontally, vertically, and diagonally.
    4. Expand the pattern until it reaches the grid edges or encounters other expanded areas.
    5. Fill in any 3x3 black areas completely surrounded by green cells.
    
    This approach creates a consistent branching structure that expands from the original green cells,
    maintaining symmetry and patterns observed in the example outputs.
    """
    seed_points = identify_seed_points(input_grid)
    output_grid = create_empty_grid(input_grid.get_dimensions())
    
    for seed in seed_points:
        apply_expansion_pattern(output_grid, seed)
    
    fill_surrounded_areas(output_grid)
    
    return output_grid

def identify_seed_points(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Identifies all green cells in the grid and returns their coordinates."""
    seed_points = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                seed_points.append((r, c))
    return seed_points

def create_empty_grid(dimensions: Tuple[int, int]) -> ColoredGrid:
    """Creates a new empty grid with the given dimensions."""
    rows, cols = dimensions
    return ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

def apply_expansion_pattern(grid: ColoredGrid, seed: Tuple[int, int]):
    """Applies the expansion pattern from a seed point."""
    x, y = seed
    directions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    for dx, dy in directions:
        expand_direction(grid, x, y, dx, dy)

def expand_direction(grid: ColoredGrid, x: int, y: int, dx: int, dy: int):
    """Expands the pattern in a given direction."""
    rows, cols = grid.get_dimensions()
    step = 0
    while True:
        cx, cy = x + dx * step, y + dy * step
        if not (0 <= cx < rows and 0 <= cy < cols):
            break
        if not fill_3x3_square(grid, cx, cy):
            break
        step += 3

def fill_3x3_square(grid: ColoredGrid, center_x: int, center_y: int) -> bool:
    """Fills a 3x3 square centered at the given coordinates. Returns False if the area is already filled."""
    filled = False
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if is_within_bounds(grid, center_x + dx, center_y + dy):
                if grid.get_cell(center_x + dx, center_y + dy) == 0:
                    grid.set_cell(center_x + dx, center_y + dy, 3)
                    filled = True
    return filled

def is_within_bounds(grid: ColoredGrid, x: int, y: int) -> bool:
    """Checks if the given coordinates are within the grid bounds."""
    rows, cols = grid.get_dimensions()
    return 0 <= x < rows and 0 <= y < cols

def fill_surrounded_areas(grid: ColoredGrid):
    """Fills in 3x3 black areas completely surrounded by green cells."""
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if is_surrounded_3x3(grid, r, c):
                fill_3x3_square(grid, r, c)

def is_surrounded_3x3(grid: ColoredGrid, center_x: int, center_y: int) -> bool:
    """Checks if a 3x3 area is completely surrounded by green cells."""
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            if dx in [-1, 0, 1] and dy in [-1, 0, 1]:
                continue
            x, y = center_x + dx, center_y + dy
            if not is_within_bounds(grid, x, y) or grid.get_cell(x, y) != 3:
                return False
    return True
