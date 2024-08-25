from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

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
    random.seed(42)  # For reproducibility
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def process_subgrid(subgrid: List[List[int]], make_blue_dominant: bool, is_middle: bool = False) -> List[List[int]]:
        dominant_color = 1 if make_blue_dominant else 2
        other_color = 2 if make_blue_dominant else 1
        
        for i in range(len(subgrid)):
            for j in range(len(subgrid[0])):
                if subgrid[i][j] != 0:
                    subgrid[i][j] = dominant_color if not is_middle else random.choice([1, 2])
        
        if not is_middle:
            # Place 2-3 cells of the other color, preferring corners and edges
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            random.shuffle(corners)
            for i, j in corners[:2]:
                subgrid[i][j] = other_color
            # Ensure clustering
            if random.choice([True, False]):
                subgrid[1][1] = other_color
        else:
            # Balance colors in the middle section
            blue_count = sum(cell == 1 for row in subgrid for cell in row)
            total_non_black = sum(cell != 0 for row in subgrid for cell in row)
            target_blue = total_non_black // 2
            while blue_count != target_blue:
                i, j = random.randint(0, 2), random.randint(0, 2)
                if subgrid[i][j] != 0:
                    if blue_count < target_blue and subgrid[i][j] == 2:
                        subgrid[i][j] = 1
                        blue_count += 1
                    elif blue_count > target_blue and subgrid[i][j] == 1:
                        subgrid[i][j] = 2
                        blue_count -= 1
        
        return subgrid
    
    # Process top section
    for i in range(0, 3):
        for j in range(0, 3):
            subgrid = [row[j*3:(j+1)*3] for row in output_grid.values[i*3:(i+1)*3]]
            processed = process_subgrid(subgrid, make_blue_dominant=True)
            for r in range(3):
                for c in range(3):
                    output_grid.values[i*3+r][j*3+c] = processed[r][c]
    
    # Process middle section
    for j in range(0, 3):
        subgrid = [row[j*3:(j+1)*3] for row in output_grid.values[3:6]]
        processed = process_subgrid(subgrid, make_blue_dominant=True, is_middle=True)
        for r in range(3):
            for c in range(3):
                output_grid.values[3+r][j*3+c] = processed[r][c]
    
    # Process bottom section
    for i in range(2, 3):
        for j in range(0, 3):
            subgrid = [row[j*3:(j+1)*3] for row in output_grid.values[i*3:(i+1)*3]]
            processed = process_subgrid(subgrid, make_blue_dominant=False)
            for r in range(3):
                for c in range(3):
                    output_grid.values[i*3+r][j*3+c] = processed[r][c]
    
    # Preserve black dividing lines
    for i in range(rows):
        output_grid.values[i][3] = output_grid.values[i][7] = 0
    for j in range(cols):
        output_grid.values[3][j] = output_grid.values[7][j] = 0
    
    return output_grid
