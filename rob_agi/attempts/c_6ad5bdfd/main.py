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

def collect_non_zeros(line: List[int]) -> List[Tuple[int, int]]:
    non_zeros = []
    for value in line:
        if value != 0:
            if not non_zeros or value != non_zeros[-1][0]:
                non_zeros.append((value, 1))
            else:
                non_zeros[-1] = (non_zeros[-1][0], non_zeros[-1][1] + 1)
    return non_zeros

def place_non_zeros(new_line: List[int], non_zeros: List[Tuple[int, int]], anchor_side: str):
    if anchor_side == "bottom" or anchor_side == "right":
        new_index = len(new_line) - sum(count for _, count in non_zeros) - 1
    else:  # left
        new_index = 1
    
    for value, count in non_zeros:
        for _ in range(count):
            new_line[new_index] = value
            new_index += 1 if anchor_side != "right" else -1

def process_line(input_line: List[int], anchor_side: str) -> List[int]:
    new_line = [0] * len(input_line)
    non_zeros = collect_non_zeros(input_line[:-1] if anchor_side == "bottom" else input_line[1:-1])
    place_non_zeros(new_line, non_zeros, anchor_side)
    
    if anchor_side == "bottom":
        new_line[-1] = input_line[-1]
    elif anchor_side == "left":
        new_line[0] = input_line[0]
    elif anchor_side == "right":
        new_line[-1] = input_line[-1]
    
    return new_line

def solve_6ad5bdfd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by moving all non-zero (colored) objects towards an anchor side.
    The anchor side is determined by finding a full line (row or column) of non-zero values.
    Objects are moved while preserving their relative positions, stacking order, and multi-color integrity.
    The anchor line remains in its original position.

    1. Identify the anchor side (bottom, left, or right).
    2. Process each line (row or column) independently:
       - Collect non-zero values, preserving multi-color objects.
       - Place these values in a new line according to the anchor side.
    3. Assemble the new lines into a transformed grid.
    4. Return the transformed grid.

    This implementation handles bottom, left, and right anchors, ensuring that objects are moved
    correctly towards the anchor side while maintaining their relative positions and
    preserving multi-color objects.
    """
    anchor_side = find_anchor_side(input_grid)
    rows, cols = input_grid.get_dimensions()
    
    if anchor_side in ["left", "right"]:
        new_values = [process_line(row, anchor_side) for row in input_grid.values]
    else:  # bottom anchor
        new_values = [process_line([row[c] for row in input_grid.values], anchor_side) for c in range(cols)]
        new_values = list(map(list, zip(*new_values)))  # Transpose the result
    
    return ColoredGrid(values=new_values)
