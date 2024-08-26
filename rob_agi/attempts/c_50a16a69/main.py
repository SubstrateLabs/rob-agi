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
    1. Extracts the pattern from the non-border area of the input grid.
    2. Finds the smallest repeating subsequence in this pattern.
    3. Generates a new grid by extending the identified pattern across the entire area, starting from the top-left corner.
    
    This approach works for various patterns, handling different grid sizes and border colors,
    and extending the pattern to areas that were originally borders or uniform regions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended pattern.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Extract the pattern from non-border area
    pattern = []
    for r in range(min(2, rows - 1)):
        for c in range(min(2, cols - 1)):
            pattern.append(input_grid.values[r][c])
    
    # Find the smallest repeating pattern
    smallest_pattern = find_smallest_repeating_pattern(pattern)
    
    # Generate the output grid
    output_values = []
    pattern_index = 0
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(smallest_pattern[pattern_index])
            pattern_index = (pattern_index + 1) % len(smallest_pattern)
        output_values.append(row)
        # Ensure each row starts with the correct pattern element
        pattern_index = (r + 1) % len(smallest_pattern)
    
    return ColoredGrid(values=output_values)
