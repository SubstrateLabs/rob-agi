from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_903d1b4a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Simplify the pattern in the input grid by removing infrequent or isolated colors
    while preserving the main structure. The solution involves:
    1. Analyzing the grid to identify main colors and structures.
    2. Processing the central area to simplify its pattern.
    3. Replacing infrequent colors throughout the grid with more common neighboring colors.
    4. Performing a final pass to remove any remaining isolated colors.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_color_frequency(grid: ColoredGrid) -> Counter:
        return Counter(color for row in grid.values for color in row)
    
    def get_central_area(grid: ColoredGrid) -> List[List[int]]:
        rows, cols = grid.get_dimensions()
        center_size = min(4, rows // 2, cols // 2)
        start_row, start_col = (rows - center_size) // 2, (cols - center_size) // 2
        return grid.extract_subgrid(start_row, start_col, center_size, center_size).values
    
    def get_neighborhood(grid: ColoredGrid, row: int, col: int) -> List[int]:
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = (row + dr) % rows, (col + dc) % cols
                neighbors.append(grid.values[r][c])
        return neighbors
    
    def get_most_frequent_color(colors: List[int]) -> int:
        return Counter(colors).most_common(1)[0][0]
    
    # Analyze the grid
    color_freq = get_color_frequency(output_grid)
    border_color = output_grid.values[0][0]
    cross_color = color_freq.most_common(2)[1][0]  # Second most common color
    
    # Process central area
    central_area = get_central_area(output_grid)
    central_freq = Counter(color for row in central_area for color in row)
    main_central_color = central_freq.most_common(1)[0][0]
    
    center_size = len(central_area)
    start_row, start_col = (rows - center_size) // 2, (cols - center_size) // 2
    for r in range(start_row, start_row + center_size):
        for c in range(start_col, start_col + center_size):
            output_grid.values[r][c] = main_central_color
    
    # Identify colors to replace
    total_squares = rows * cols
    colors_to_replace = [color for color, freq in color_freq.items() 
                         if freq < total_squares * 0.05 and color not in [border_color, cross_color, main_central_color]]
    
    # Replace infrequent colors
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in colors_to_replace:
                neighbors = get_neighborhood(output_grid, r, c)
                new_color = get_most_frequent_color([color for color in neighbors if color not in colors_to_replace])
                output_grid.values[r][c] = new_color
    
    # Final pass to remove isolated colors
    for r in range(rows):
        for c in range(cols):
            neighbors = get_neighborhood(output_grid, r, c)
            if output_grid.values[r][c] not in neighbors:
                output_grid.values[r][c] = get_most_frequent_color(neighbors)
    
    return output_grid
