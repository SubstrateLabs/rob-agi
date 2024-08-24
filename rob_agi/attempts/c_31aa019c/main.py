from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Optional

def solve_31aa019c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 31aa019c challenge by finding the unique non-zero number in the input grid,
    creating a 3x3 square centered on that number's position, and returning the result.
    
    The unique number is placed at the center of the 3x3 square, surrounded by 2's.
    If no unique number is found, returns a 10x10 grid of zeros.
    """
    def find_unique_number(grid: List[List[int]]) -> Optional[Tuple[int, int, int]]:
        flat_grid = [num for row in grid for num in row if num != 0]
        counter = Counter(flat_grid)
        unique = [num for num, count in counter.items() if count == 1]
        if not unique:
            return None
        unique_num = unique[0]
        for i, row in enumerate(grid):
            for j, num in enumerate(row):
                if num == unique_num:
                    return (unique_num, i, j)
        return None

    def create_output_grid(unique_num: int, row: int, col: int) -> List[List[int]]:
        output = [[0] * 10 for _ in range(10)]
        for i in range(max(0, row-1), min(10, row+2)):
            for j in range(max(0, col-1), min(10, col+2)):
                if i == row and j == col:
                    output[i][j] = unique_num
                else:
                    output[i][j] = 2
        return output

    input_values = input_grid.values
    result = find_unique_number(input_values)
    
    if result is None:
        return ColoredGrid(values=[[0] * 10 for _ in range(10)])
    
    unique_num, row, col = result
    output_grid = create_output_grid(unique_num, row, col)
    
    return ColoredGrid(values=output_grid)
