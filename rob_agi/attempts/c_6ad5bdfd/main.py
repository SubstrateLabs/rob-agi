from rob_agi.colored_grid import ColoredGrid

def find_anchor_side(grid: ColoredGrid) -> str:
    if all(grid.values[-1]):
        return "bottom"
    if all(row[0] for row in grid.values):
        return "left"
    if all(row[-1] for row in grid.values):
        return "right"
    raise ValueError("No anchor side found")

def solve_6ad5bdfd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero (colored) objects towards an anchor side.
    The anchor side is determined by finding a full line (row or column) of non-zero values.
    Objects are moved while preserving their relative positions and stacking order.
    The anchor line remains in its original position.

    1. Identify the anchor side (bottom, left, or right).
    2. Create a new empty grid with the same dimensions.
    3. Move objects towards the anchor side, maintaining their order and vertical alignment.
    4. Copy the anchor line to the new grid.
    5. Return the transformed grid.
    """
    anchor_side = find_anchor_side(input_grid)
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    if anchor_side == "bottom":
        for c in range(cols):
            non_zeros = [input_grid.values[r][c] for r in range(rows) if input_grid.values[r][c] != 0]
            for r, value in enumerate(reversed(non_zeros), start=rows-len(non_zeros)):
                new_grid.values[r][c] = value
        new_grid.values[-1] = input_grid.values[-1]  # Copy anchor line
    
    elif anchor_side in ["left", "right"]:
        for c in range(cols):
            non_zeros = [(r, input_grid.values[r][c]) for r in range(rows) if input_grid.values[r][c] != 0]
            if anchor_side == "left":
                for i, (r, value) in enumerate(non_zeros):
                    new_grid.values[r][i] = value
            else:  # right
                for i, (r, value) in enumerate(reversed(non_zeros), start=cols-len(non_zeros)):
                    new_grid.values[r][i] = value
        
        # Copy anchor line
        if anchor_side == "left":
            for r in range(rows):
                new_grid.values[r][0] = input_grid.values[r][0]
        else:  # right
            for r in range(rows):
                new_grid.values[r][-1] = input_grid.values[r][-1]
    
    return new_grid
