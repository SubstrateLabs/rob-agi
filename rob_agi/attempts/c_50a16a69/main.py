from rob_agi.colored_grid import ColoredGrid

def find_smallest_repeating_pattern(sequence):
    for i in range(1, len(sequence) // 2 + 1):
        if len(sequence) % i == 0:
            if sequence[:i] * (len(sequence) // i) == sequence:
                return sequence[:i]
    return sequence

def solve_50a16a69(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the core repeating pattern and extending it to cover the entire grid.
    
    The function performs the following steps:
    1. Extracts the pattern from the top-left quarter of the input grid.
    2. Finds the smallest repeating subsequence in both horizontal and vertical directions.
    3. Generates a new grid by extending the identified patterns across the entire area, starting from the top-left corner.
    
    This approach works for various patterns, including checkerboard patterns, handling different grid sizes and border colors,
    and extending the pattern to areas that were originally borders or uniform regions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended pattern.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Extract the pattern from the top-left quarter
    pattern_rows = rows // 2
    pattern_cols = cols // 2
    
    # Find the horizontal pattern
    horizontal_pattern = find_smallest_repeating_pattern(input_grid.values[0][:pattern_cols])
    
    # Find the vertical pattern
    vertical_pattern = find_smallest_repeating_pattern([input_grid.values[r][0] for r in range(pattern_rows)])
    
    # Generate the output grid
    output_values = []
    for r in range(rows):
        row = []
        for c in range(cols):
            color = horizontal_pattern[c % len(horizontal_pattern)]
            if r % 2 == 1:  # Alternate the pattern for odd rows
                color = horizontal_pattern[(c + 1) % len(horizontal_pattern)]
            row.append(color)
        output_values.append(row)
    
    return ColoredGrid(values=output_values)
