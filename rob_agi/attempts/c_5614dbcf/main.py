from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_5614dbcf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x9 input grid into a 3x3 output grid by extracting the center elements of each 3x3 subgrid.
    
    The input grid is divided into nine 3x3 subgrids. The function extracts the center element
    from each subgrid and places it in the corresponding position in the output grid.
    If the center element is 5 (gray), it is replaced with the most frequent non-zero, non-gray color in that subgrid.
    If there are no other colors, it is replaced with 0 (black).
    
    Args:
    input_grid (ColoredGrid): A 9x9 input grid
    
    Returns:
    ColoredGrid: A 3x3 output grid containing the processed center elements of each input subgrid
    """
    output = [[0 for _ in range(3)] for _ in range(3)]
    
    for i in range(3):
        for j in range(3):
            # Calculate the center position of each 3x3 subgrid in the input
            center_row = i * 3 + 1
            center_col = j * 3 + 1
            
            # Extract the center element
            center_value = input_grid.get_cell(center_row, center_col)
            
            if center_value == 5:  # If the center is gray (5)
                # Get all values in the 3x3 subgrid
                subgrid_values = [
                    input_grid.get_cell(r, c)
                    for r in range(i*3, (i+1)*3)
                    for c in range(j*3, (j+1)*3)
                ]
                # Count non-zero, non-gray values
                color_counts = Counter(v for v in subgrid_values if v not in [0, 5])
                # Replace with most frequent non-zero, non-gray color, or 0 if no other colors
                center_value = color_counts.most_common(1)[0][0] if color_counts else 0
            
            # Place the processed center element in the output grid
            output[i][j] = center_value
    
    return ColoredGrid(values=output)
