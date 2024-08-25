from rob_agi.colored_grid import ColoredGrid

def solve_fe9372f3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following pattern:
    1. Create a base grid with yellow (4) squares every 4 cells, connected by sky blue (8) lines.
    2. Add blue (1) diagonal lines from corners to opposite corners.
    3. Copy the red (2) cross from the input grid.
    4. Ensure all other cells remain black (0).
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Create base grid pattern
    for r in range(rows):
        for c in range(cols):
            if r % 4 == 0 and c % 4 == 0:
                output_grid.set_cell(r, c, 4)  # Yellow
            elif r % 4 == 0 or c % 4 == 0:
                output_grid.set_cell(r, c, 8)  # Sky blue

    # Add diagonal blue lines
    for i in range(max(rows, cols)):
        if i < rows and i < cols:
            output_grid.set_cell(i, i, 1)  # Top-left to bottom-right
            output_grid.set_cell(i, cols-1-i, 1)  # Top-right to bottom-left
        if rows-1-i >= 0 and i < cols:
            output_grid.set_cell(rows-1-i, i, 1)  # Bottom-left to top-right
        if i < rows and cols-1-i >= 0:
            output_grid.set_cell(i, cols-1-i, 1)  # Bottom-right to top-left

    # Copy red cross from input grid
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 2:
                output_grid.set_cell(r, c, 2)

    return output_grid
