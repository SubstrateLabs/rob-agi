from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid:
    1. The output grid size is the square of the input grid size.
    2. For odd-sized inputs, fill the middle row and column with the input pattern.
    3. For even-sized inputs, fill the entire perimeter with the input pattern and replicate the input in the corners.
    4. For even-sized inputs, also fill the rows and columns adjacent to the middle with the input pattern.
    5. The rest of the grid remains filled with zeros (black).
    """
    n = len(input_grid.values)
    output_size = n * n
    output_values = [[0 for _ in range(output_size)] for _ in range(output_size)]
    
    if n % 2 == 1:  # Odd-sized input
        mid = output_size // 2
        for i in range(output_size):
            output_values[mid][i] = input_grid.values[n//2][i % n]
            output_values[i][mid] = input_grid.values[i % n][n//2]
    else:  # Even-sized input
        mid = output_size // 2
        for i in range(output_size):
            # Fill top, bottom, left, and right edges
            output_values[0][i] = input_grid.values[0][i % n]
            output_values[output_size - 1][i] = input_grid.values[n - 1][i % n]
            output_values[i][0] = input_grid.values[i % n][0]
            output_values[i][output_size - 1] = input_grid.values[i % n][n - 1]
            
            # Fill rows and columns adjacent to the middle
            output_values[mid - 1][i] = input_grid.values[n//2 - 1][i % n]
            output_values[mid][i] = input_grid.values[n//2][i % n]
            output_values[i][mid - 1] = input_grid.values[i % n][n//2 - 1]
            output_values[i][mid] = input_grid.values[i % n][n//2]
        
        # Fill the corners with complete copies of the input
        for i in range(n):
            for j in range(n):
                output_values[i][j] = input_grid.values[i][j]  # Top-left
                output_values[i][output_size - n + j] = input_grid.values[i][j]  # Top-right
                output_values[output_size - n + i][j] = input_grid.values[i][j]  # Bottom-left
                output_values[output_size - n + i][output_size - n + j] = input_grid.values[i][j]  # Bottom-right
    
    return ColoredGrid(values=output_values)
