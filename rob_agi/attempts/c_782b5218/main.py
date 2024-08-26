from rob_agi.colored_grid import ColoredGrid

def solve_782b5218(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a horizontal striped pattern based on the following rules:
    
    1. Identifies unique colors in the input grid.
    2. Sorts the colors in ascending order.
    3. Creates horizontal stripes for each color, with the following special cases:
       - The top 3 rows are always filled with the lowest color (usually black, 0).
       - If red (2) is present, it forms a single row stripe in the middle.
       - The remaining colors fill the bottom rows, with the most frequent color 
         (excluding 0 and 2) filling all remaining rows.
    
    Returns a new ColoredGrid with the transformed horizontal striped pattern.
    """
    rows, cols = input_grid.get_dimensions()
    unique_colors = sorted(set(color for row in input_grid.values for color in row))
    return create_horizontal_stripes(rows, cols, unique_colors, input_grid)

def create_horizontal_stripes(rows: int, cols: int, sorted_colors: list, input_grid: ColoredGrid) -> ColoredGrid:
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Fill top 3 rows with the lowest color
    for r in range(3):
        new_values[r] = [sorted_colors[0]] * cols
    
    # Handle red stripe if present
    if 2 in sorted_colors:
        red_row = 3
        new_values[red_row] = [2] * cols
        sorted_colors.remove(2)
    else:
        red_row = -1
    
    # Find the most frequent color (excluding 0 and 2)
    color_freq = input_grid.get_color_frequencies()
    most_frequent_color = max((c for c in color_freq if c not in (0, 2)), key=color_freq.get)
    
    # Fill remaining rows
    for r in range(red_row + 1, rows):
        new_values[r] = [most_frequent_color] * cols
    
    return ColoredGrid(values=new_values)
