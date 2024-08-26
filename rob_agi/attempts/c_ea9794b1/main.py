from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by analyzing 4x4 regions and color patterns.
    
    The transformation process involves:
    1. Analyzing global color distribution and patterns in the input grid.
    2. Processing 4x4 regions of the input grid to determine each output cell.
    3. Prioritizing green (3) and preserving distinctive patterns of other colors.
    4. Balancing local color information with global patterns and distribution.
    5. Applying special rules for color boundaries and transitions.
    
    Args:
    input_grid (ColoredGrid): A 10x10 input grid

    Returns:
    ColoredGrid: A 5x5 output grid
    """
    if input_grid.get_dimensions() != (10, 10):
        raise ValueError("Input grid must be 10x10")

    def analyze_global_colors(grid):
        flat_grid = [cell for row in grid.values for cell in row]
        return Counter(flat_grid)

    def process_region(region):
        flat_region = [cell for row in region for cell in row]
        color_count = Counter(flat_region)
        
        # Prioritize green
        if color_count[3] >= 3:
            return 3
        
        # Check for distinctive patterns
        if color_count[9] >= 4:  # Brown
            return 9
        if color_count[8] >= 4:  # Sky
            return 8
        
        # Handle color transitions
        unique_colors = set(flat_region) - {0}
        if len(unique_colors) == 2 and 3 in unique_colors:
            return 3
        
        # Use the most frequent non-black color
        for color, count in color_count.most_common():
            if color != 0:
                return color
        
        return 0  # Default to black if no other color is present

    global_colors = analyze_global_colors(input_grid)
    output_values = []

    for i in range(0, 10, 2):
        row = []
        for j in range(0, 10, 2):
            region = [input_grid.values[i+di][j:j+4] for di in range(4)]
            color = process_region(region)
            row.append(color)
        output_values.append(row)

    # Post-processing to balance global color distribution
    total_cells = 25
    green_count = sum(row.count(3) for row in output_values)
    if green_count < total_cells // 5 and global_colors[3] > 0:
        for i in range(5):
            for j in range(5):
                if output_values[i][j] == 0 and green_count < total_cells // 5:
                    output_values[i][j] = 3
                    green_count += 1

    return ColoredGrid(values=output_values)
