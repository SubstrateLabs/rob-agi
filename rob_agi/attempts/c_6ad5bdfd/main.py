from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def find_anchor_side(grid: ColoredGrid) -> str:
    if all(grid.values[-1]):
        return "bottom"
    if all(row[0] for row in grid.values):
        return "left"
    if all(row[-1] for row in grid.values):
        return "right"
    raise ValueError("No anchor side found")

def collect_non_zeros(column: List[int]) -> List[Tuple[int, int]]:
    non_zeros = []
    for value in column:
        if value != 0:
            if not non_zeros or value != non_zeros[-1][0]:
                non_zeros.append((value, 1))
            else:
                non_zeros[-1] = (non_zeros[-1][0], non_zeros[-1][1] + 1)
    return non_zeros

def place_non_zeros(new_column: List[int], non_zeros: List[Tuple[int, int]], anchor_side: str):
    if anchor_side == "bottom":
        new_row = len(new_column) - sum(count for _, count in non_zeros)
    else:
        new_row = 0
    
    for value, count in non_zeros:
        for _ in range(count):
            new_column[new_row] = value
            new_row += 1

def process_column(input_column: List[int], anchor_side: str) -> List[int]:
    new_column = [0] * len(input_column)
    non_zeros = collect_non_zeros(input_column[:-1] if anchor_side == "bottom" else input_column)
    place_non_zeros(new_column, non_zeros, anchor_side)
    
    if anchor_side == "bottom":
        new_column[-1] = input_column[-1]
    elif anchor_side in ["left", "right"]:
        new_column[0 if anchor_side == "left" else -1] = input_column[0 if anchor_side == "left" else -1]
    
    return new_column

def solve_6ad5bdfd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero (colored) objects towards an anchor side.
    The anchor side is determined by finding a full line (row or column) of non-zero values.
    Objects are moved while preserving their relative positions, stacking order, and multi-color integrity.
    The anchor line remains in its original position.

    1. Identify the anchor side (bottom, left, or right).
    2. Process each column independently:
       - Collect non-zero values, preserving multi-color objects.
       - Place these values in a new column according to the anchor side.
    3. Assemble the new columns into a transformed grid.
    4. Return the transformed grid.

    This implementation handles bottom, left, and right anchors, ensuring that objects are moved
    correctly towards the anchor side while maintaining their relative positions within each column
    and preserving multi-color objects.
    """
    anchor_side = find_anchor_side(input_grid)
    rows, cols = input_grid.get_dimensions()
    
    if anchor_side in ["left", "right"]:
        new_values = [process_column([row[c] for row in input_grid.values], anchor_side) for c in range(cols)]
        new_values = list(map(list, zip(*new_values)))  # Transpose the result
    else:  # bottom anchor
        new_values = [process_column(input_grid.values[r], anchor_side) for r in range(rows)]
    
    return ColoredGrid(values=new_values)
