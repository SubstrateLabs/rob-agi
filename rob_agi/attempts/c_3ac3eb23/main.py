from rob_agi.colored_grid import ColoredGrid

def solve_3ac3eb23(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by repeating the pattern of non-zero colors from the first row.
    
    The transformation follows these rules:
    1. Analyze the first row to identify non-zero colors and their positions.
    2. Create an output grid of the same dimensions as the input.
    3. For each non-zero color found in the first row:
       a. On even rows (including the first row):
          - Place the color in its original column.
       b. On odd rows:
          - Place the color in the column to the left of its original position (if not at the left edge).
          - Place the color in the column to the right of its original position (if not at the right edge).
    4. Repeat this pattern for all rows in the output grid.
    """
    height, width = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Find non-zero colors in the first row
    colors = [(col, input_grid.get_cell(0, col)) for col in range(width) if input_grid.get_cell(0, col) != 0]
    
    for row in range(height):
        for col, color in colors:
            if row % 2 == 0:
                # Even rows: place color in original column
                output.set_cell(row, col, color)
            else:
                # Odd rows: place color in adjacent columns
                if col > 0:
                    output.set_cell(row, col - 1, color)
                if col < width - 1:
                    output.set_cell(row, col + 1, color)
    
    return output
