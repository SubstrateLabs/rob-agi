from rob_agi.colored_grid import ColoredGrid

def solve_1c56ad9f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a zigzag pattern on the vertical edges of shapes.
    Odd-numbered rows extend one unit to the left, while even-numbered rows extend one unit to the right.
    This creates a wave-like effect on the vertical sides of shapes while preserving their overall structure.
    """
    result = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    for row in range(input_grid.num_rows):
        leftmost = rightmost = -1
        for col in range(input_grid.num_cols):
            if input_grid[row][col] != 0:
                if leftmost == -1:
                    leftmost = col
                rightmost = col
        
        if leftmost == -1:  # No non-zero elements in this row
            result.values[row] = input_grid[row].copy()
        else:
            if (row + 1) % 2 == 1:  # Odd-numbered row
                if leftmost > 0:
                    result.values[row][leftmost-1] = input_grid[row][leftmost]
                for col in range(leftmost, input_grid.num_cols):
                    result.values[row][col] = input_grid[row][col]
            else:  # Even-numbered row
                for col in range(leftmost, rightmost + 1):
                    result.values[row][col] = input_grid[row][col]
                if rightmost < input_grid.num_cols - 1:
                    result.values[row][rightmost+1] = input_grid[row][rightmost]
    
    return result
