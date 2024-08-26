from rob_agi.colored_grid import ColoredGrid

def solve_90347967(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rotating it 90 degrees clockwise and moving non-black cells to the top-right corner.
    
    1. Scans the input grid from left to right, bottom to top.
    2. Creates new columns (which will become rows after rotation) preserving the order of non-zero elements.
    3. Pads shorter columns with zeros at the beginning to ensure right alignment.
    4. Transposes the resulting grid to complete the 90-degree clockwise rotation.
    """
    rows, cols = input_grid.get_dimensions()
    new_columns = []

    # Scan and create new columns
    for col in range(cols):
        new_column = [input_grid.values[row][col] for row in range(rows-1, -1, -1) if input_grid.values[row][col] != 0]
        if new_column:
            new_columns.append(new_column)

    # Find the maximum length of new columns
    max_length = max(len(col) for col in new_columns) if new_columns else 0

    # Pad shorter columns with zeros at the beginning
    padded_columns = [([0] * (max_length - len(col))) + col for col in new_columns]

    # Create the new grid by transposing padded_columns
    new_grid_values = list(map(list, zip(*padded_columns)))

    # Pad the new grid with zeros if necessary to maintain the original dimensions
    while len(new_grid_values) < rows:
        new_grid_values.append([0] * cols)
    
    return ColoredGrid(values=new_grid_values)
