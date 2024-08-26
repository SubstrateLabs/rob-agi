from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_4acc7107(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Vertically flips all non-black shapes
    2. Moves shapes to the bottom of the grid
    3. Consolidates disconnected shapes of the same color
    4. Maintains relative horizontal positions and color adjacency where possible
    
    The transformation preserves the overall structure of the shapes while
    reorganizing them at the bottom of the grid.
    """
    rows, cols = input_grid.get_dimensions()
    color_coords = defaultdict(list)
    
    # Step 1: Analyze the input grid
    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color != 0:
                color_coords[color].append((r, c))
    
    # Step 2: Create a new empty grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Helper functions
    def vertical_flip(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(rows - 1 - r, c) for r, c in coords]
    
    def center_of_mass(coords: List[Tuple[int, int]]) -> float:
        return sum(c for _, c in coords) / len(coords)
    
    def find_bottom_position(flipped_coords: List[Tuple[int, int]]) -> int:
        max_row = max(r for r, _ in flipped_coords)
        for offset in range(rows):
            if all(0 <= r + offset < rows for r, _ in flipped_coords):
                return offset
        return 0
    
    # Step 3: Process each color
    for color, coords in sorted(color_coords.items(), key=lambda x: -min(r for r, _ in x[1])):
        flipped_coords = vertical_flip(coords)
        original_com = center_of_mass(coords)
        flipped_com = center_of_mass(flipped_coords)
        
        # Find the lowest possible position
        bottom_offset = find_bottom_position(flipped_coords)
        
        # Adjust horizontal position
        horizontal_shift = round(original_com - flipped_com)
        
        # Place the shape in the new grid
        for r, c in flipped_coords:
            new_r = r + bottom_offset
            new_c = max(0, min(cols - 1, c + horizontal_shift))
            if 0 <= new_r < rows and 0 <= new_c < cols:
                new_grid[new_r][new_c] = color
    
    return ColoredGrid(values=new_grid)
