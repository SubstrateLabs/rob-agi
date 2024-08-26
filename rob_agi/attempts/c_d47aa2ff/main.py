from rob_agi.colored_grid import ColoredGrid

def solve_d47aa2ff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 10x21 input grid into a 10x10 output grid by:
    1. Extracting the left 10x10 portion of the input grid.
    2. Analyzing color pairs on both sides of the central gray line.
    3. Adding new blue (1) and red (2) dots based on color pair occurrences.
    4. Placing new dots in available positions on the right side of the output grid.

    Args:
    input_grid (ColoredGrid): A 10x21 grid with a central gray (5) line.

    Returns:
    ColoredGrid: A 10x10 grid with the transformation applied.
    """
    # Step 1: Extract the Input
    output_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for i in range(10):
        for j in range(10):
            output_grid.values[i][j] = input_grid.values[i][j]
    
    # Step 2: Analyze Color Pairs
    left_colors = {}
    right_colors = {}
    for i in range(10):
        for j in range(10):
            if input_grid.values[i][j] != 0:
                left_colors[input_grid.values[i][j]] = left_colors.get(input_grid.values[i][j], 0) + 1
        for j in range(11, 21):
            if input_grid.values[i][j] != 0:
                right_colors[input_grid.values[i][j]] = right_colors.get(input_grid.values[i][j], 0) + 1
    
    # Step 3: Determine New Dots
    new_dots = []
    for color in set(left_colors.keys()) & set(right_colors.keys()):
        if left_colors[color] >= 1 and right_colors[color] >= 1:
            new_dots.append((1, 1))  # Add a blue dot
            if left_colors[color] >= 2 and right_colors[color] >= 2:
                new_dots.append((2, 1))  # Add a red dot
    
    # Step 4: Place New Dots
    new_dots.sort()  # Ensure blue dots are placed before red dots
    available_positions = [(i, j) for i in range(10) for j in range(5, 10) if output_grid.values[i][j] == 0]
    
    for dot in new_dots:
        if available_positions:
            pos = available_positions.pop(0)
            output_grid.values[pos[0]][pos[1]] = dot[0]
    
    # Step 5: Finalize Output
    return output_grid
