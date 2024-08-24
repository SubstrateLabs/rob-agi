from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    
    def is_plus_shape(grid, row, col):
        if grid.values[row][col] != 1:  # Center must be blue
            return False
        # Check four adjacent cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if not (0 <= row + dr < rows and 0 <= col + dc < cols) or grid.values[row + dr][col + dc] != 1:
                return False
        return True
    
    # First pass: Mark plus shapes
    marking_grid = [[False for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            if is_plus_shape(input_grid, row, col):
                marking_grid[row][col] = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    marking_grid[row + dr][col + dc] = True
    
    # Second pass: Transform the grid
    output_grid = input_grid.deep_copy()
    for row in range(rows):
        for col in range(cols):
            if marking_grid[row][col]:
                output_grid.values[row][col] = 2  # Red
            elif input_grid.values[row][col] == 1:
                output_grid.values[row][col] = 3  # Green
    
    return output_grid
