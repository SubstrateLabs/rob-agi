from rob_agi.colored_grid import ColoredGrid

def identify_border(grid):
    rows, cols = grid.get_dimensions()
    border_colors = set(grid.values[0] + grid.values[-1] + [row[0] for row in grid.values] + [row[-1] for row in grid.values])
    return border_colors

def find_core_sequence(grid, border_colors):
    sequence = []
    rows, cols = grid.get_dimensions()
    for i in range(min(rows, cols)):
        color = grid.values[i][i]
        if color not in border_colors and color not in sequence:
            sequence.append(color)
        if len(sequence) == 2:  # We only need the first 2 colors of the pattern
            break
    return sequence

def determine_start_color(grid, sequence, border_colors):
    for i in range(min(grid.get_dimensions())):
        color = grid.values[i][i]
        if color not in border_colors:
            return color
    return sequence[0]  # Fallback to first color in sequence

def generate_output_grid(input_grid, sequence, start_color):
    rows, cols = input_grid.get_dimensions()
    output_values = []
    start_index = sequence.index(start_color)
    for i in range(rows):
        row = []
        for j in range(cols):
            color = sequence[(start_index + i + j) % len(sequence)]
            row.append(color)
        output_values.append(row)
    return ColoredGrid(values=output_values)

def solve_50a16a69(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the core repeating sequence and extending it to cover the entire grid.
    
    The function performs the following steps:
    1. Identifies the border colors of the input grid.
    2. Finds the core sequence of colors from the non-border area.
    3. Determines the starting color for the output pattern.
    4. Generates a new grid by extending the identified sequence across the entire area, starting with the correct color.
    
    This approach works for various patterns, handling different grid sizes, border colors,
    and extending the pattern to areas that were originally borders or uniform regions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended pattern.
    """
    border_colors = identify_border(input_grid)
    core_sequence = find_core_sequence(input_grid, border_colors)
    start_color = determine_start_color(input_grid, core_sequence, border_colors)
    return generate_output_grid(input_grid, core_sequence, start_color)
