from rob_agi.colored_grid import ColoredGrid

def solve_90347967(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rotating it 90 degrees clockwise and moving non-black cells to the top-right corner.
    
    1. Scans the input grid from bottom to top, left to right.
    2. Groups non-zero elements by their new column index after rotation.
    3. Sorts these groups based on their original column order.
    4. Places the sorted groups into a new grid, starting from the top-right corner and moving left.
    5. Within each group, elements maintain their relative vertical positions.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    column_groups = []

    # Scan and Group
    for original_col in range(cols):
        group = []
        for original_row in range(rows - 1, -1, -1):  # bottom to top
            if input_grid.values[original_row][original_col] != 0:
                new_col = original_row
                group.append((input_grid.values[original_row][original_col], original_col))
        if group:
            column_groups.append(group)

    # Sort Groups
    column_groups.sort(key=lambda g: g[0][1])  # Sort based on original column

    # Place Groups in New Grid
    current_col = cols - 1
    for group in column_groups:
        for row, (value, _) in enumerate(group):
            new_grid.values[row][current_col] = value
        current_col -= 1

    return new_grid
