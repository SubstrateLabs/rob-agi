from rob_agi.colored_grid import ColoredGrid

def solve_fd4b2b02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following pattern:
    1. Identify the non-zero shape and its color.
    2. Create a symmetrical pattern with corner elements of the same color as the input shape.
    3. Place elements of the complementary color between the corner elements.
    4. Preserve the original input shape or integrate its essence into the pattern.
    5. Add small accents of the complementary color if space allows.
    6. Ensure symmetry by mirroring the pattern in all four quadrants.

    Args:
    input_grid (ColoredGrid): The input grid to transform

    Returns:
    ColoredGrid: The transformed output grid
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find the non-zero shape
    main_color = None
    shape_coords = []
    min_row, min_col, max_row, max_col = rows, cols, 0, 0
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                main_color = input_grid.values[r][c]
                shape_coords.append((r, c))
                min_row, min_col = min(min_row, r), min(min_col, c)
                max_row, max_col = max(max_row, r), max(max_col, c)

    if not shape_coords:
        return output_grid

    # Define colors and dimensions
    comp_color = 3 if main_color == 6 else 6
    shape_width = max_col - min_col + 1
    shape_height = max_row - min_row + 1

    # Calculate pattern dimensions
    corner_size = min(3, min(shape_width, shape_height)) if rows <= 20 else min(4, min(shape_width, shape_height))
    side_size = max(shape_width, shape_height)

    # Create the basic pattern
    pattern_size = 2 * corner_size + side_size
    pattern = [[0 for _ in range(pattern_size)] for _ in range(pattern_size)]

    # Place corner elements
    for r, c in [(0, 0), (0, pattern_size-corner_size), (pattern_size-corner_size, 0), (pattern_size-corner_size, pattern_size-corner_size)]:
        for dr in range(corner_size):
            for dc in range(corner_size):
                pattern[r + dr][c + dc] = main_color

    # Place side elements
    for r, c in [(0, corner_size), (corner_size, 0), (pattern_size-corner_size, corner_size), (corner_size, pattern_size-corner_size)]:
        for dr in range(side_size):
            for dc in range(side_size):
                pattern[r + dr][c + dc] = comp_color

    # Center the pattern in the output grid
    offset_r = (rows - pattern_size) // 2
    offset_c = (cols - pattern_size) // 2

    # Apply the pattern to the output grid
    for r in range(pattern_size):
        for c in range(pattern_size):
            if 0 <= offset_r + r < rows and 0 <= offset_c + c < cols:
                output_grid.values[offset_r + r][offset_c + c] = pattern[r][c]

    # Preserve the original shape
    center_r, center_c = rows // 2, cols // 2
    for r, c in shape_coords:
        dr, dc = r - center_r, c - center_c
        if abs(dr) < pattern_size // 2 and abs(dc) < pattern_size // 2:
            output_grid.values[r][c] = main_color

    # Add accent in bottom-right if space allows
    if rows > pattern_size + 2 and cols > pattern_size + 2:
        for r in range(rows - 2, rows):
            for c in range(cols - 3, cols):
                output_grid.values[r][c] = comp_color

    # Ensure symmetry
    for r in range(rows):
        for c in range(cols):
            if r < rows // 2 and c >= cols // 2:
                output_grid.values[r][c] = output_grid.values[r][cols - c - 1]
            elif r >= rows // 2:
                output_grid.values[r][c] = output_grid.values[rows - r - 1][c]

    return output_grid
