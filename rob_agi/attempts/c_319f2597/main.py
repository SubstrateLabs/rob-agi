from rob_agi.colored_grid import ColoredGrid

def solve_319f2597(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a central vertical black stripe and at least two horizontal black lines.
    
    The transformation follows these steps:
    1. Create a 2-column wide vertical black stripe in the center of the grid.
    2. Identify rows with existing black squares (0) in the input.
    3. Create horizontal black lines in rows with existing black squares, or choose rows if none exist.
    4. Ensure at least two horizontal black lines are created.
    5. Handle intersections and adjacent squares to maintain consistency.
    6. Preserve the leftmost and rightmost columns, and top and bottom rows where possible.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the specified pattern applied.
    """
    # Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Create the central vertical black stripe
    center_left = cols // 2 - 1
    center_right = cols // 2
    for row in range(rows):
        output_grid.set_cell(row, center_left, 0)
        output_grid.set_cell(row, center_right, 0)
    
    # Identify rows with black squares in the input
    black_rows = [row for row in range(rows) if 0 in input_grid.values[row]]
    
    # Create horizontal black lines
    if not black_rows or len(black_rows) < 2:
        # Choose rows based on the lowest sum of values if not enough black rows
        row_sums = [sum(input_grid.values[row]) for row in range(rows)]
        additional_rows = sorted(range(rows), key=lambda x: row_sums[x])[:max(2, 2 - len(black_rows))]
        black_rows.extend(additional_rows)
        black_rows = sorted(set(black_rows))[:2]  # Ensure at least 2 unique rows
    
    for row in black_rows:
        for col in range(2, cols - 2):
            output_grid.set_cell(row, col, 0)
    
    # Handle intersections and adjacent squares
    for row in range(rows):
        for col in range(cols):
            if output_grid.get_cell(row, col) == 0:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols and input_grid.get_cell(nr, nc) == 0:
                        output_grid.set_cell(nr, nc, 0)
    
    return output_grid
