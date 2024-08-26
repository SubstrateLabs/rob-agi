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
    4. Fill in any gaps in the resulting pattern.
    
    This approach creates a consistent branching structure that expands from the original green cells,
    maintaining symmetry and patterns observed in the example outputs.
    """
    seed_points = identify_seed_points(input_grid)
    output_grid = create_empty_grid(input_grid.get_dimensions())
    
    for seed in seed_points:
        apply_expansion_pattern(output_grid, seed)
    
    fill_gaps(output_grid)
    
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
    fill_3x3_square(grid, x, y)
    
    for offset in range(3, max(grid.get_dimensions()), 3):
        fill_3x3_square(grid, x + offset, y)
        fill_3x3_square(grid, x - offset, y)
        fill_3x3_square(grid, x, y + offset)
        fill_3x3_square(grid, x, y - offset)
        fill_3x3_square(grid, x + offset, y + offset)
        fill_3x3_square(grid, x - offset, y - offset)
        fill_3x3_square(grid, x + offset, y - offset)
        fill_3x3_square(grid, x - offset, y + offset)

def fill_3x3_square(grid: ColoredGrid, center_x: int, center_y: int):
    """Fills a 3x3 square centered at the given coordinates."""
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if is_within_bounds(grid, center_x + dx, center_y + dy):
                grid.set_cell(center_x + dx, center_y + dy, 3)

def is_within_bounds(grid: ColoredGrid, x: int, y: int) -> bool:
    """Checks if the given coordinates are within the grid bounds."""
    rows, cols = grid.get_dimensions()
    return 0 <= x < rows and 0 <= y < cols

def fill_gaps(grid: ColoredGrid):
    """Fills in gaps in the pattern."""
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0 and all_neighbors_green(grid, r, c):
                grid.set_cell(r, c, 3)

def all_neighbors_green(grid: ColoredGrid, x: int, y: int) -> bool:
    """Checks if all 8 neighbors of a cell are green."""
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if is_within_bounds(grid, nx, ny) and grid.get_cell(nx, ny) != 3:
                return False
    return True
