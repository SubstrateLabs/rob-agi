from rob_agi.colored_grid import ColoredGrid

def get_core_sequence(grid):
    sequence = []
    rows, cols = grid.get_dimensions()
    for i in range(min(rows, cols)):
        color = grid.values[i][i]
        if color not in sequence and not is_border_color(grid, color):
            sequence.append(color)
        if len(sequence) == 3:  # We only need the first 3 colors of the pattern
            break
    return sequence

def is_border_color(grid, color):
    rows, cols = grid.get_dimensions()
    return color in grid.values[-1] or any(row[-1] == color for row in grid.values)

def create_output_grid(input_grid, sequence):
    rows, cols = input_grid.get_dimensions()
    output_values = []
    for i in range(rows):
        row = []
        for j in range(cols):
            color = sequence[(i + j) % len(sequence)]
            row.append(color)
        output_values.append(row)
    return ColoredGrid(values=output_values)

def solve_50a16a69(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the core repeating sequence and extending it to cover the entire grid.
    
    The function performs the following steps:
    1. Identifies the core sequence from the top-left diagonal of the input grid, ignoring border colors.
    2. Creates a new grid by extending the identified sequence across the entire area.
    
    This approach works for various patterns, handling different grid sizes, border colors,
    and extending the pattern to areas that were originally borders or uniform regions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended pattern.
    """
    core_sequence = get_core_sequence(input_grid)
    return create_output_grid(input_grid, core_sequence)
