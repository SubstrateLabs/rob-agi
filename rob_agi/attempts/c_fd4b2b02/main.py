from rob_agi.colored_grid import ColoredGrid

def solve_fd4b2b02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following pattern:
    1. Identify the central non-zero shape and its color.
    2. Create a frame structure with corner elements of the same color as the central shape.
    3. Place square elements of the complementary color between the corner elements.
    4. Preserve the original input shape in its position.
    5. Add an edge line at the bottom-right corner.
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

    # Calculate dimensions for corner elements and square elements
    corner_width = shape_width
    corner_height = min(3, shape_height) if rows <= 16 else min(4, shape_height)
    square_size = max(shape_width, shape_height)

    # Place corner elements
    for r, c in [(0, 0), (0, cols-corner_width), (rows-corner_height, 0), (rows-corner_height, cols-corner_width)]:
        for dr in range(corner_height):
            for dc in range(corner_width):
                if r + dr < rows and c + dc < cols:
                    output_grid.values[r + dr][c + dc] = main_color

    # Place square elements
    square_positions = [
        (0, corner_width),
        (0, cols - corner_width - square_size),
        (rows - square_size, corner_width),
        (rows - square_size, cols - corner_width - square_size),
        (corner_height, 0),
        (rows - corner_height - square_size, 0),
        (corner_height, cols - square_size),
        (rows - corner_height - square_size, cols - square_size)
    ]
    for r, c in square_positions:
        for dr in range(square_size):
            for dc in range(square_size):
                if 0 <= r + dr < rows and 0 <= c + dc < cols:
                    output_grid.values[r + dr][c + dc] = comp_color

    # Preserve the original input shape
    for r, c in shape_coords:
        output_grid.values[r][c] = main_color

    # Add edge line
    if shape_width > shape_height:
        for c in range(cols - square_size, cols):
            output_grid.values[rows - 1][c] = main_color
    else:
        for r in range(rows - square_size, rows):
            output_grid.values[r][cols - 1] = main_color

    # Fill remaining spaces and ensure symmetry
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 0:
                if r < rows // 2 and c < cols // 2:
                    output_grid.values[r][c] = main_color
                elif r < rows // 2 and c >= cols // 2:
                    output_grid.values[r][c] = output_grid.values[r][cols - c - 1]
                elif r >= rows // 2 and c < cols // 2:
                    output_grid.values[r][c] = output_grid.values[rows - r - 1][c]
                else:
                    output_grid.values[r][c] = output_grid.values[rows - r - 1][cols - c - 1]

    return output_grid
