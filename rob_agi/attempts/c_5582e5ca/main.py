from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_5582e5ca(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling it entirely with the dominant color.
    The dominant color is the most frequent color in the input grid.
    In case of a tie, the highest-valued color among the tied colors is chosen.
    """
    # Flatten the 2D grid into a 1D list
    flat_grid = [cell for row in input_grid.values for cell in row]
    
    # Count color frequencies
    color_counts = Counter(flat_grid)
    
    # Determine the dominant color
    max_frequency = max(color_counts.values())
    dominant_color = max(color for color, count in color_counts.items() if count == max_frequency)
    
    # Generate the output grid
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    output_grid = [[dominant_color for _ in range(cols)] for _ in range(rows)]
    
    return ColoredGrid(values=output_grid)
