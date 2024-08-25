from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_dc2aa30b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by inverting color dominance between top and bottom sections,
    while maintaining a balanced mix in the middle section. The solution:
    1. Divides the grid into 3x3 sub-grids
    2. Makes the top section predominantly blue with some red clusters
    3. Makes the bottom section predominantly red with some blue clusters
    4. Balances the middle section with a mix of blue and red
    5. Preserves the black (0) dividing lines
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def process_subgrid(subgrid: List[List[int]], make_blue_dominant: bool) -> List[List[int]]:
        blue_count = sum(cell == 1 for row in subgrid for cell in row)
        red_count = sum(cell == 2 for row in subgrid for cell in row)
        total_cells = len(subgrid) * len(subgrid[0])
        
        dominant_color = 1 if make_blue_dominant else 2
        other_color = 2 if make_blue_dominant else 1
        
        # Convert most cells to the dominant color
        for i in range(len(subgrid)):
            for j in range(len(subgrid[0])):
                if subgrid[i][j] != 0:
                    subgrid[i][j] = dominant_color
        
        # Place 2-3 cells of the other color, preferring corners and edges
        corners = [(0, 0), (0, len(subgrid[0])-1), (len(subgrid)-1, 0), (len(subgrid)-1, len(subgrid[0])-1)]
        for i, j in corners[:2]:
            subgrid[i][j] = other_color
        
        # Ensure the other color cells are clustered
        if len(subgrid) > 2 and len(subgrid[0]) > 2:
            subgrid[1][1] = other_color
        
        return subgrid
    
    # Process top section
    for i in range(0, 3):
        for j in range(0, 3):
            if i != 1 or j != 1:  # Skip the middle subgrid
                subgrid = [row[j*3:(j+1)*3] for row in output_grid.values[i*3:(i+1)*3]]
                processed = process_subgrid(subgrid, make_blue_dominant=True)
                for r in range(3):
                    for c in range(3):
                        output_grid.values[i*3+r][j*3+c] = processed[r][c]
    
    # Process bottom section
    for i in range(2, 3):
        for j in range(0, 3):
            if i != 1 or j != 1:  # Skip the middle subgrid
                subgrid = [row[j*3:(j+1)*3] for row in output_grid.values[i*3:(i+1)*3]]
                processed = process_subgrid(subgrid, make_blue_dominant=False)
                for r in range(3):
                    for c in range(3):
                        output_grid.values[i*3+r][j*3+c] = processed[r][c]
    
    # Process middle section
    for j in range(0, 3):
        subgrid = [row[j*3:(j+1)*3] for row in output_grid.values[3:6]]
        blue_count = sum(cell == 1 for row in subgrid for cell in row)
        red_count = sum(cell == 2 for row in subgrid for cell in row)
        total_cells = len(subgrid) * len(subgrid[0])
        
        if blue_count / total_cells > 0.7:
            target_red = int(total_cells * 0.4)
            for i in range(len(subgrid)):
                for j in range(len(subgrid[0])):
                    if subgrid[i][j] == 1 and red_count < target_red:
                        subgrid[i][j] = 2
                        red_count += 1
        elif red_count / total_cells > 0.7:
            target_blue = int(total_cells * 0.4)
            for i in range(len(subgrid)):
                for j in range(len(subgrid[0])):
                    if subgrid[i][j] == 2 and blue_count < target_blue:
                        subgrid[i][j] = 1
                        blue_count += 1
        
        for r in range(3):
            for c in range(3):
                output_grid.values[3+r][j*3+c] = subgrid[r][c]
    
    # Preserve black dividing lines
    for i in range(rows):
        output_grid.values[i][3] = output_grid.values[i][7] = 0
    for j in range(cols):
        output_grid.values[3][j] = output_grid.values[7][j] = 0
    
    return output_grid
