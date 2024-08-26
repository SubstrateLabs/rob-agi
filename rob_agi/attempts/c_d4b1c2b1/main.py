from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_d4b1c2b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expands a colored grid based on the following rules:
    1. If the grid is uniform (all cells have the same color), return the input grid unchanged.
    2. Otherwise, expand each cell into a square region, where the size of the square is determined by the complexity of the input grid.
    The complexity is measured by the number of unique color pairs between adjacent cells.
    """
    if is_uniform(input_grid):
        return input_grid

    complexity = analyze_complexity(input_grid)
    expansion_factor = get_expansion_factor(complexity)
    
    expanded_grid = create_expanded_grid(input_grid, expansion_factor)
    fill_expanded_grid(input_grid, expanded_grid, expansion_factor)

    return expanded_grid

def is_uniform(grid: ColoredGrid) -> bool:
    """Check if all cells in the grid have the same color."""
    first_color = grid.get_cell(0, 0)
    rows, cols = grid.get_dimensions()
    return all(grid.get_cell(r, c) == first_color for r in range(rows) for c in range(cols))

def analyze_complexity(grid: ColoredGrid) -> int:
    """Analyze the complexity of the grid by counting unique color pairs."""
    unique_pairs: Set[Tuple[int, int]] = set()
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if c < cols - 1:
                right_color = grid.get_cell(r, c + 1)
                unique_pairs.add((min(color, right_color), max(color, right_color)))
            if r < rows - 1:
                bottom_color = grid.get_cell(r + 1, c)
                unique_pairs.add((min(color, bottom_color), max(color, bottom_color)))
    return len(unique_pairs)

def get_expansion_factor(complexity: int) -> int:
    """Determine the expansion factor based on the grid complexity."""
    if complexity <= 3:
        return 2
    elif complexity <= 5:
        return 3
    else:
        return 4

def create_expanded_grid(original_grid: ColoredGrid, expansion_factor: int) -> ColoredGrid:
    """Create an expanded grid based on the original grid and expansion factor."""
    rows, cols = original_grid.get_dimensions()
    new_rows, new_cols = rows * expansion_factor, cols * expansion_factor
    return ColoredGrid(values=[[0 for _ in range(new_cols)] for _ in range(new_rows)])

def fill_expanded_grid(original_grid: ColoredGrid, expanded_grid: ColoredGrid, expansion_factor: int) -> None:
    """Fill the expanded grid with colors from the original grid."""
    rows, cols = original_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = original_grid.get_cell(r, c)
            for dr in range(expansion_factor):
                for dc in range(expansion_factor):
                    expanded_grid.set_cell(r * expansion_factor + dr, c * expansion_factor + dc, color)
