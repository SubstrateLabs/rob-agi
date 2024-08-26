from rob_agi.colored_grid import ColoredGrid

def find_anchor_side(grid: ColoredGrid) -> str:
    if all(grid.values[-1]):
        return "bottom"
    if all(row[0] for row in grid.values):
        return "left"
    if all(row[-1] for row in grid.values):
        return "right"
    raise ValueError("No anchor side found")

def process_column(input_grid: ColoredGrid, new_grid: ColoredGrid, col: int, anchor_side: str):
    rows, cols = input_grid.get_dimensions()
    non_zeros = []
    start_row = 0 if anchor_side != "bottom" else 0
    end_row = rows if anchor_side != "bottom" else rows - 1
    
    # Collect non-zero values
    for r in range(start_row, end_row):
        if input_grid.values[r][col] != 0:
            if not non_zeros or input_grid.values[r][col] != non_zeros[-1][0]:
                non_zeros.append((input_grid.values[r][col], 1))
            else:
                non_zeros[-1] = (non_zeros[-1][0], non_zeros[-1][1] + 1)
    
    # Place non-zero values in the new grid
    if anchor_side == "bottom":
        new_row = rows - 1 - sum(count for _, count in non_zeros)
        for value, count in non_zeros:
            for _ in range(count):
                new_grid.values[new_row][col] = value
                new_row += 1
    elif anchor_side == "left":
        new_row = 0
        for value, count in non_zeros:
            for _ in range(count):
                new_grid.values[new_row][col] = value
                new_row += 1
    elif anchor_side == "right":
        new_row = rows - sum(count for _, count in non_zeros)
        for value, count in non_zeros:
            for _ in range(count):
                new_grid.values[new_row][col] = value
                new_row += 1

def solve_6ad5bdfd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero (colored) objects towards an anchor side.
    The anchor side is determined by finding a full line (row or column) of non-zero values.
    Objects are moved while preserving their relative positions and stacking order.
    The anchor line remains in its original position.

    1. Identify the anchor side (bottom, left, or right).
    2. Create a new empty grid with the same dimensions.
    3. Process each column independently:
       - Collect non-zero values, preserving multi-color objects.
       - Place these values in the new grid according to the anchor side.
    4. Copy the anchor line to the new grid.
    5. Fill any remaining spaces with zeros.
    6. Return the transformed grid.

    This implementation handles bottom, left, and right anchors, ensuring that objects are moved
    correctly towards the anchor side while maintaining their relative positions within each column
    and preserving multi-color objects.
    """
    anchor_side = find_anchor_side(input_grid)
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    for c in range(cols):
        process_column(input_grid, new_grid, c, anchor_side)
    
    # Copy anchor line
    if anchor_side == "bottom":
        new_grid.values[-1] = input_grid.values[-1]
    elif anchor_side == "left":
        for r in range(rows):
            new_grid.values[r][0] = input_grid.values[r][0]
    elif anchor_side == "right":
        for r in range(rows):
            new_grid.values[r][-1] = input_grid.values[r][-1]
    
    return new_grid
