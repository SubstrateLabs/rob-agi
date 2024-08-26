from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def find_empty_cell(grid: ColoredGrid) -> Optional[Tuple[int, int]]:
    for r in range(4):
        for c in range(4):
            if grid.values[r][c] == 0:
                return (r, c)
    return None

def is_valid_placement(grid: ColoredGrid, row: int, col: int, num: int) -> bool:
    # Check row
    if num in grid.values[row]:
        return False
    
    # Check column
    if num in [grid.values[r][col] for r in range(4)]:
        return False
    
    return True

def get_possible_numbers(grid: ColoredGrid, row: int, col: int) -> List[int]:
    used = set(grid.values[row] + [grid.values[r][col] for r in range(4)])
    return [num for num in range(1, 5) if num not in used]

def solve_backtrack(grid: ColoredGrid) -> bool:
    cell = find_empty_cell(grid)
    if not cell:
        return True  # Grid is filled
    
    row, col = cell
    for num in get_possible_numbers(grid, row, col):
        if is_valid_placement(grid, row, col, num):
            grid.values[row][col] = num
            if solve_backtrack(grid):
                return True
            grid.values[row][col] = 0  # Backtrack
    
    return False  # No valid number found

def is_valid_solution(grid: ColoredGrid) -> bool:
    for i in range(4):
        if set(grid.values[i]) != set(range(1, 5)) or \
           set(grid.values[r][i] for r in range(4)) != set(range(1, 5)):
            return False
    return True

def solve_4cd1b7b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 4x4 grid by filling in black (0) squares with numbers 1-4,
    ensuring each row and column contains exactly one of each number 1-4.
    
    The function uses a backtracking algorithm to find a valid solution.
    It preserves non-zero numbers in their original positions and fills
    in the black (0) squares to create a valid Latin square.
    
    Args:
    input_grid (ColoredGrid): The input 4x4 grid with some filled and some black squares.
    
    Returns:
    ColoredGrid: The completed grid with all squares filled to form a valid Latin square.
    
    Raises:
    ValueError: If the input grid is not 4x4, contains invalid numbers, or no solution exists.
    """
    if input_grid.get_dimensions() != (4, 4):
        raise ValueError("Input grid must be 4x4")
    
    if any(any(val not in range(5) for val in row) for row in input_grid.values):
        raise ValueError("Input grid contains invalid numbers")
    
    grid = input_grid.deep_copy()
    
    if solve_backtrack(grid):
        if is_valid_solution(grid):
            return grid
        else:
            raise ValueError("Found solution is invalid")
    else:
        raise ValueError("No solution exists")
