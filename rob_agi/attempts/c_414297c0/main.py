from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_414297c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the largest contiguous colored region as the background,
    preserving all other colored elements, and arranging them in a compact manner.
    
    1. Identifies the largest contiguous colored region (background).
    2. Preserves all other colored elements.
    3. Creates a new grid with the background color.
    4. Places preserved elements in their relative positions.
    5. Optimizes the grid by removing unnecessary background-only rows/columns.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Step 1: Identify the largest contiguous region
    background_color, _ = find_largest_region(input_grid)
    
    # Step 2: Create a list of elements to preserve
    elements_to_preserve = find_elements_to_preserve(input_grid, background_color)
    
    # Step 3 & 4: Determine dimensions and create output grid
    output_grid = create_output_grid(elements_to_preserve, background_color)
    
    # Step 5: Place preserved elements in the output grid
    place_preserved_elements(output_grid, elements_to_preserve)
    
    # Step 6: Optimize the output grid
    optimized_grid = optimize_grid(output_grid)
    
    # Step 7: Create and return the final ColoredGrid object
    return ColoredGrid(values=optimized_grid)

def find_largest_region(grid: ColoredGrid) -> Tuple[int, int]:
    largest_color = 0
    largest_size = 0
    for color in range(10):  # 0 to 9
        regions = grid.find_connected_regions(color)
        if regions:
            size = max(len(region) for region in regions)
            if size > largest_size:
                largest_size = size
                largest_color = color
    return largest_color, largest_size

def find_elements_to_preserve(grid: ColoredGrid, background_color: int) -> List[Tuple[int, int, int]]:
    elements = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != background_color and grid.values[r][c] != 0:
                elements.append((grid.values[r][c], r, c))
    return elements

def create_output_grid(elements: List[Tuple[int, int, int]], background_color: int) -> List[List[int]]:
    if not elements:
        return [[background_color]]
    min_row = min(e[1] for e in elements)
    max_row = max(e[1] for e in elements)
    min_col = min(e[2] for e in elements)
    max_col = max(e[2] for e in elements)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    return [[background_color for _ in range(width)] for _ in range(height)]

def place_preserved_elements(grid: List[List[int]], elements: List[Tuple[int, int, int]]):
    min_row = min(e[1] for e in elements)
    min_col = min(e[2] for e in elements)
    for color, r, c in elements:
        grid[r - min_row][c - min_col] = color

def optimize_grid(grid: List[List[int]]) -> List[List[int]]:
    # Remove empty rows from top and bottom
    while grid and all(cell == grid[0][0] for cell in grid[0]):
        grid.pop(0)
    while grid and all(cell == grid[-1][0] for cell in grid[-1]):
        grid.pop()
    
    # Remove empty columns from left and right
    while grid and all(row[0] == grid[0][0] for row in grid):
        for row in grid:
            row.pop(0)
    while grid and all(row[-1] == grid[0][-1] for row in grid):
        for row in grid:
            row.pop()
    
    return grid
