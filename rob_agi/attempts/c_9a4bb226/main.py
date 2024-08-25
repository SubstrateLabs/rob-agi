from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_9a4bb226(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Finds a 3x3 subgrid within the input grid that contains exactly three colors,
    where one color forms an 'L' shape and the other two colors fill the remaining spaces.
    
    The function scans the input grid for all possible 3x3 subgrids, checks if they meet
    the criteria, and returns the first valid subgrid found as a new ColoredGrid.
    
    If no valid subgrid is found, returns None.
    """
    def is_valid_subgrid(subgrid: List[List[int]]) -> bool:
        colors = set(color for row in subgrid for color in row if color != 0)
        if len(colors) != 3:
            return False
        
        l_shapes = [
            [(0,0), (1,0), (2,0), (2,1), (2,2)],
            [(0,0), (0,1), (0,2), (1,2), (2,2)],
            [(0,0), (0,1), (0,2), (1,0), (2,0)],
            [(0,0), (1,0), (2,0), (0,1), (0,2)]
        ]
        
        for color in colors:
            for l_shape in l_shapes:
                if all(subgrid[r][c] == color for r, c in l_shape):
                    other_colors = colors - {color}
                    if all(subgrid[r][c] in other_colors 
                           for r in range(3) for c in range(3) 
                           if (r,c) not in l_shape):
                        return True
        return False

    rows, cols = input_grid.get_dimensions()
    for i in range(rows - 2):
        for j in range(cols - 2):
            subgrid = [row[j:j+3] for row in input_grid.values[i:i+3]]
            if is_valid_subgrid(subgrid):
                return ColoredGrid(values=subgrid)
    
    return None
