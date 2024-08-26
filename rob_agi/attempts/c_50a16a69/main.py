from rob_agi.colored_grid import ColoredGrid

def identify_border_colors(grid):
    rows, cols = grid.get_dimensions()
    return set(grid.values[0] + grid.values[-1] + [row[0] for row in grid.values] + [row[-1] for row in grid.values])

def identify_non_border_colors(grid, border_colors):
    return set(color for row in grid.values for color in row if color not in border_colors)

def determine_color_sequence(grid, non_border_colors):
    rows, cols = grid.get_dimensions()
    color_positions = {color: (rows * cols, 0) for color in non_border_colors}
    for r in range(rows):
        for c in range(cols):
            color = grid.values[r][c]
            if color in non_border_colors:
                position = r * cols + c
                if position < color_positions[color][0]:
                    color_positions[color] = (position, color_positions[color][1])
                color_positions[color] = (color_positions[color][0], color_positions[color][1] + 1)
    return [color for color, _ in sorted(color_positions.items(), key=lambda x: (x[1][0], -x[1][1]))]

def generate_output_grid(input_grid, sequence):
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
    1. Identifies the border colors of the input grid.
    2. Identifies the non-border colors.
    3. Determines the sequence of non-border colors based on their first occurrence and frequency.
    4. Generates a new grid by extending the identified sequence across the entire area.
    
    This approach works for various patterns, handling different grid sizes, border colors,
    and extending the pattern to areas that were originally borders or uniform regions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended pattern.
    """
    border_colors = identify_border_colors(input_grid)
    non_border_colors = identify_non_border_colors(input_grid, border_colors)
    
    if not non_border_colors:
        non_border_colors = set(color for row in input_grid.values for color in row)
    
    sequence = determine_color_sequence(input_grid, non_border_colors)
    
    if not sequence:
        return input_grid  # If no sequence is found, return the input grid unchanged
    
    return generate_output_grid(input_grid, sequence)
