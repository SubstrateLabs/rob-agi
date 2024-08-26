from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by processing 2x2 subgrids.
    
    For each 2x2 subgrid:
    1. If green (3) appears more than once, use green.
    2. Otherwise, use the most frequent color.
    3. In case of a tie:
       - Prefer non-black over black.
       - If still tied, use position-based priority: top-left, top-right, bottom-left, bottom-right.
    4. The chosen color for each 2x2 subgrid is placed in the corresponding position of the 5x5 output grid.
    
    Args:
    input_grid (ColoredGrid): A 10x10 input grid

    Returns:
    ColoredGrid: A 5x5 output grid
    """
    if input_grid.get_dimensions() != (10, 10):
        raise ValueError("Input grid must be 10x10")

    def process_subgrid(subgrid):
        flat_subgrid = [cell for row in subgrid for cell in row]
        color_count = Counter(flat_subgrid)
        
        if color_count[3] > 1:
            return 3
        
        max_count = max(color_count.values())
        max_colors = [color for color, count in color_count.items() if count == max_count]
        
        if len(max_colors) == 1:
            return max_colors[0]
        
        if 0 in max_colors and len(max_colors) > 1:
            max_colors.remove(0)
        
        for position in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            if subgrid[position[0]][position[1]] in max_colors:
                return subgrid[position[0]][position[1]]
        
        return 0  # This should never happen, but just in case

    output_values = []
    for i in range(0, 10, 2):
        row = []
        for j in range(0, 10, 2):
            subgrid = [input_grid.values[i+di][j:j+2] for di in range(2)]
            row.append(process_subgrid(subgrid))
        output_values.append(row)

    return ColoredGrid(values=output_values)
