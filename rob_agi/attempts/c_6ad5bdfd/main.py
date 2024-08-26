from rob_agi.colored_grid import ColoredGrid

def find_anchor_side(grid: ColoredGrid) -> str:
    if all(grid.values[-1]):
        return "bottom"
    if all(row[0] for row in grid.values):
        return "left"
    if all(row[-1] for row in grid.values):
        return "right"
    raise ValueError("No anchor side found")

def process_bottom_anchor(input_grid: ColoredGrid, new_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for c in range(cols):
        non_zeros = [input_grid.values[r][c] for r in range(rows-1) if input_grid.values[r][c] != 0]
        for r, value in enumerate(reversed(non_zeros), start=rows-1-len(non_zeros)):
            new_grid.values[r][c] = value
    new_grid.values[-1] = input_grid.values[-1]  # Copy anchor line

def process_left_anchor(input_grid: ColoredGrid, new_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for c in range(1, cols):
        non_zeros = [input_grid.values[r][c] for r in range(rows) if input_grid.values[r][c] != 0]
        for r, value in enumerate(non_zeros):
            new_grid.values[r][c] = value
    for r in range(rows):
        new_grid.values[r][0] = input_grid.values[r][0]  # Copy anchor line

def process_right_anchor(input_grid: ColoredGrid, new_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        non_zeros = [input_grid.values[r][c] for c in range(cols-1) if input_grid.values[r][c] != 0]
        for c, value in enumerate(non_zeros, start=cols-1-len(non_zeros)):
            new_grid.values[r][c] = value
        new_grid.values[r][-1] = input_grid.values[r][-1]  # Copy anchor line

def solve_6ad5bdfd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero (colored) objects towards an anchor side.
    The anchor side is determined by finding a full line (row or column) of non-zero values.
    Objects are moved while preserving their relative positions and stacking order.
    The anchor line remains in its original position.

    1. Identify the anchor side (bottom, left, or right).
    2. Create a new empty grid with the same dimensions.
    3. Copy the anchor line to the new grid.
    4. Move objects towards the anchor side:
       - For bottom anchor: move down, preserving columns
       - For left anchor: move left, preserving columns
       - For right anchor: move right, preserving columns
    5. Maintain the vertical order of elements within each column.
    6. Fill any remaining spaces with zeros.
    7. Return the transformed grid.

    This implementation handles bottom, left, and right anchors, ensuring that objects are moved
    correctly towards the anchor side while maintaining their relative positions within each column.
    """
    anchor_side = find_anchor_side(input_grid)
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    if anchor_side == "bottom":
        process_bottom_anchor(input_grid, new_grid)
    elif anchor_side == "left":
        process_left_anchor(input_grid, new_grid)
    elif anchor_side == "right":
        process_right_anchor(input_grid, new_grid)
    
    return new_grid
