from rob_agi.colored_grid import ColoredGrid

def solve_fd4b2b02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following pattern:
    1. Identify the non-zero shape in the input grid.
    2. Create a 4x4 alternating color pattern based on the input shape's color.
    3. Repeat the 4x4 pattern across the grid, extending to the edges.
    4. Adjust the bottom-right corner if necessary.
    5. Preserve the original input shape in its position.
    6. Ensure symmetry in the output grid.

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

    # Define colors for the pattern
    main_color = non_zero_color
    secondary_color = 3 if main_color == 6 else 6

    # Create and apply the 4x4 pattern
    pattern = [
        [main_color, secondary_color, main_color, main_color],
        [main_color, 0, main_color, main_color],
        [main_color, secondary_color, main_color, main_color],
        [main_color, secondary_color, main_color, main_color]
    ]

    for r in range(0, rows, 4):
        for c in range(0, cols, 4):
            for dr in range(4):
                for dc in range(4):
                    if r + dr < rows and c + dc < cols:
                        output_grid.values[r + dr][c + dc] = pattern[dr][dc]

    # Adjust bottom-right corner
    if rows % 4 != 0 or cols % 4 != 0:
        output_grid.values[rows-1][cols-1] = secondary_color

    # Preserve the original input shape
    for r, c in shape_coords:
        output_grid.values[r][c] = non_zero_color

    # Ensure symmetry
    for r in range(rows):
        output_grid.values[r][0] = output_grid.values[r][-1] = secondary_color
    for c in range(cols):
        output_grid.values[0][c] = output_grid.values[-1][c] = secondary_color

    return output_grid
