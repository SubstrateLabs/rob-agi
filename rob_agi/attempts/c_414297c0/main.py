from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_414297c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by selecting the most frequent non-black color as background,
    preserving all other colored elements and formations, and arranging them
    while maintaining their relative positions.
    
    1. Analyzes the input grid to identify the most frequent non-black color for background.
    2. Identifies the rectangular area of this background color in the input.
    3. Creates a new grid based on the background rectangle with a one-cell border.
    4. Maps and places non-background elements in the new grid, maintaining relative positions.
    5. Adjusts element positions to ensure background color separation.
    6. Removes unnecessary background-only rows and columns.
    7. Performs final adjustments to ensure proper element placement and border.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Step 1: Analyze input and select background color
    background_color = select_background_color(input_grid)
    
    # Step 2: Identify background rectangle
    bg_rect = find_background_rectangle(input_grid, background_color)
    
    # Step 3: Create initial output grid
    output_grid = create_initial_grid(bg_rect, background_color)
    
    # Step 4 & 5: Map and place elements
    place_elements(input_grid, output_grid, bg_rect, background_color)
    
    # Step 6: Remove unnecessary space
    optimized_grid = remove_unnecessary_space(output_grid, background_color)
    
    # Step 7: Perform final adjustments
    final_grid = final_adjustments(optimized_grid, background_color)
    
    # Create and return the final ColoredGrid object
    return ColoredGrid(values=final_grid)

def select_background_color(grid: ColoredGrid) -> int:
    """Selects the most frequent non-black color as the background."""
    color_counts = grid.get_color_frequencies()
    if 0 in color_counts:
        del color_counts[0]  # Remove black (0) from consideration
    return max(color_counts, key=color_counts.get) if color_counts else 1  # Default to blue if no other colors

def find_background_rectangle(grid: ColoredGrid, background_color: int) -> Tuple[int, int, int, int]:
    """Finds the largest rectangle of background color."""
    rows, cols = grid.get_dimensions()
    top = bottom = left = right = None
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == background_color:
                if top is None:
                    top = r
                bottom = r
                if left is None or c < left:
                    left = c
                if right is None or c > right:
                    right = c
    return (top, left, bottom, right)

def create_initial_grid(bg_rect: Tuple[int, int, int, int], background_color: int) -> List[List[int]]:
    """Creates the initial output grid based on the background rectangle."""
    top, left, bottom, right = bg_rect
    height = bottom - top + 3  # Add 2 for border
    width = right - left + 3  # Add 2 for border
    return [[background_color for _ in range(width)] for _ in range(height)]

def place_elements(input_grid: ColoredGrid, output_grid: List[List[int]], bg_rect: Tuple[int, int, int, int], background_color: int):
    """Maps and places non-background elements in the output grid."""
    top, left, bottom, right = bg_rect
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = input_grid.values[r][c]
            if color != 0 and color != background_color:
                new_r = r - top + 1
                new_c = c - left + 1
                if 0 <= new_r < len(output_grid) and 0 <= new_c < len(output_grid[0]):
                    output_grid[new_r][new_c] = color

def remove_unnecessary_space(grid: List[List[int]], background_color: int) -> List[List[int]]:
    """Removes unnecessary background-only rows and columns."""
    rows, cols = len(grid), len(grid[0])
    row_keep = [False] * rows
    col_keep = [False] * cols
    
    # Mark rows and columns to keep
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background_color:
                row_keep[r] = True
                col_keep[c] = True
    
    # Ensure border rows and columns are kept
    row_keep[0] = row_keep[-1] = True
    col_keep[0] = col_keep[-1] = True
    
    # Create new grid with only necessary rows and columns
    new_grid = [[grid[r][c] for c in range(cols) if col_keep[c]] for r in range(rows) if row_keep[r]]
    return new_grid

def final_adjustments(grid: List[List[int]], background_color: int) -> List[List[int]]:
    """Performs final adjustments to ensure proper element placement and border."""
    rows, cols = len(grid), len(grid[0])
    
    # Ensure border
    for r in range(rows):
        grid[r][0] = grid[r][-1] = background_color
    for c in range(cols):
        grid[0][c] = grid[-1][c] = background_color
    
    # Ensure elements are not touching
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] != background_color:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for nr, nc in neighbors:
                    if grid[nr][nc] != background_color:
                        # Move the element if it's touching another non-background element
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            if grid[r+dr][c+dc] == background_color:
                                grid[r+dr][c+dc], grid[r][c] = grid[r][c], background_color
                                break
    
    return grid
