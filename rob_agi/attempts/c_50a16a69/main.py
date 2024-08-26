from rob_agi.colored_grid import ColoredGrid

def get_core_sequence(grid):
    sequence = []
    rows, cols = grid.get_dimensions()
    i, j = 0, 0
    while i < rows - 1 and j < cols - 1 and len(sequence) < 4:
        color = grid.values[i][j]
        if color not in sequence and not is_border_color(grid, color):
            sequence.append(color)
        i += 1
        j += 1
    return sequence

def is_border_color(grid, color):
    rows, cols = grid.get_dimensions()
    return all(grid.values[i][cols-1] == color for i in range(rows)) or \
           all(grid.values[rows-1][j] == color for j in range(cols))

def rotate_sequence(sequence):
    return sequence[1:] + [sequence[0]]

def create_output_grid(input_grid, rotated_sequence):
    rows, cols = input_grid.get_dimensions()
    output_values = []
    for i in range(rows):
        row = []
        for j in range(cols):
            color = rotated_sequence[(i + j) % len(rotated_sequence)]
            row.append(color)
        output_values.append(row)
    return ColoredGrid(values=output_values)

def solve_50a16a69(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the core repeating sequence and extending it to cover the entire grid.
    
    The function performs the following steps:
    1. Identifies the core sequence from the top-left diagonal of the input grid, ignoring border colors.
    2. Rotates the sequence by moving the first color to the end.
    3. Creates a new grid by extending the rotated sequence across the entire area.
    
    This approach works for various patterns, handling different grid sizes, border colors,
    and extending the pattern to areas that were originally borders or uniform regions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended pattern.
    """
    core_sequence = get_core_sequence(input_grid)
    rotated_sequence = rotate_sequence(core_sequence)
    return create_output_grid(input_grid, rotated_sequence)
