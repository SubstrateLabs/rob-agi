from rob_agi.colored_grid import ColoredGrid

def solve_fd4b2b02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following pattern:
    1. Identify the non-zero shape in the input grid.
    2. Create a unit cell pattern based on the input shape's dimensions and color.
    3. Repeat the unit cell pattern across the grid, extending to the edges.
    4. Preserve the original input shape in its position.
    5. Ensure symmetry in the output grid.

    The unit cell pattern consists of:
    - Corner squares of the appropriate color (3 or 6)
    - Horizontal and vertical lines at 1/4 and 3/4 of the cell dimensions
    - Color 3 (green) for horizontal lines if input is 6, and vice versa

    Args:
    input_grid (ColoredGrid): The input grid to transform

    Returns:
    ColoredGrid: The transformed output grid
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find the non-zero shape
    non_zero_color = None
    shape_coords = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                non_zero_color = input_grid.values[r][c]
                shape_coords.append((r, c))

    if not shape_coords:
        return output_grid

    # Determine unit cell size
    shape_width = max(c for _, c in shape_coords) - min(c for _, c in shape_coords) + 1
    shape_height = max(r for r, _ in shape_coords) - min(r for r, _ in shape_coords) + 1
    unit_size = max(shape_width, shape_height)

    # Define colors for the pattern
    main_color = non_zero_color
    secondary_color = 3 if main_color == 6 else 6

    # Create and apply the pattern
    for r in range(0, rows, unit_size):
        for c in range(0, cols, unit_size):
            # Corner squares
            for corner_r, corner_c in [(0, 0), (0, unit_size-1), (unit_size-1, 0), (unit_size-1, unit_size-1)]:
                if r + corner_r < rows and c + corner_c < cols:
                    output_grid.values[r + corner_r][c + corner_c] = main_color

            # Horizontal lines
            for line_r in [unit_size // 4, 3 * unit_size // 4]:
                if r + line_r < rows:
                    for dc in range(unit_size):
                        if c + dc < cols:
                            output_grid.values[r + line_r][c + dc] = secondary_color

            # Vertical lines
            for line_c in [unit_size // 4, 3 * unit_size // 4]:
                if c + line_c < cols:
                    for dr in range(unit_size):
                        if r + dr < rows:
                            output_grid.values[r + dr][c + line_c] = main_color

    # Preserve the original input shape
    for r, c in shape_coords:
        output_grid.values[r][c] = non_zero_color

    return output_grid
