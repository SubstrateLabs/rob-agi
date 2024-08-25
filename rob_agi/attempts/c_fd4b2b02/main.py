from rob_agi.colored_grid import ColoredGrid

def solve_fd4b2b02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following pattern:
    1. Identify the non-zero shape and its color in the input grid.
    2. Create a frame structure using the complementary color.
    3. Fill the space between frame elements with the main color.
    4. Preserve the original input shape in its position.
    5. Ensure symmetry and extend the pattern to the grid edges.

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
    frame_color = 3 if main_color == 6 else 6
    shape_width = max_col - min_col + 1
    shape_height = max_row - min_row + 1

    # Create frame structure
    corner_width = max(2, shape_width)
    corner_height = min(3, shape_height)

    # Place corner elements
    for r, c in [(0, 0), (0, cols-corner_width), (rows-corner_height, 0), (rows-corner_height, cols-corner_width)]:
        for dr in range(corner_height):
            for dc in range(corner_width):
                if r + dr < rows and c + dc < cols:
                    output_grid.values[r + dr][c + dc] = frame_color

    # Place edge elements
    for r in range(corner_height, rows - corner_height):
        output_grid.values[r][0] = output_grid.values[r][-1] = frame_color
    for c in range(corner_width, cols - corner_width):
        output_grid.values[0][c] = output_grid.values[-1][c] = frame_color

    # Fill space between frame elements
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = main_color

    # Preserve the original input shape
    for r, c in shape_coords:
        output_grid.values[r][c] = main_color

    # Ensure symmetry
    for r in range(rows):
        if output_grid.values[r][0] != output_grid.values[r][-1]:
            output_grid.values[r][0] = output_grid.values[r][-1] = frame_color
    for c in range(cols):
        if output_grid.values[0][c] != output_grid.values[-1][c]:
            output_grid.values[0][c] = output_grid.values[-1][c] = frame_color

    return output_grid
