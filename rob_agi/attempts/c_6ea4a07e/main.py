from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_6ea4a07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing the most frequent non-zero color with black (0) and
    the black cells with a new color calculated based on the replaced color.
    The new color is determined by the formula: new_color = (replaced_color - 1) % 8 + 1,
    which ensures the new color is always between 1 and 8.
    Any other colors in the grid remain unchanged.
    """
    # Count non-zero colors
    color_counter = Counter(cell for row in input_grid.values for cell in row if cell != 0)
    
    # Find the most frequent non-zero color
    replaced_color = color_counter.most_common(1)[0][0] if color_counter else 0
    
    # Calculate new color
    new_color = (replaced_color - 1) % 8 + 1
    
    # Create new grid
    new_grid = [
        [0 if cell == replaced_color else new_color if cell == 0 else cell for cell in row]
        for row in input_grid.values
    ]
    
    # Return new ColoredGrid
    return ColoredGrid(values=new_grid)
