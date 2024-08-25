from rob_agi.colored_grid import ColoredGrid
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

def solve_a096bf4d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Analyze the grid to identify special colors and their positions in each 5x5 section.
    2. Propagate special colors within their respective rows and columns.
    3. Fill remaining spaces with the most common interior color.
    4. Handle color replacements (6 with 1, 4 with other special colors if present).
    5. Ensure consistency across all sections while preserving the border structure.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Step 1: Analyze the grid
    section_info = analyze_grid(input_grid)
    
    # Step 2 & 3: Propagate special colors and fill remaining spaces
    for r in range(0, rows, 5):
        for c in range(0, cols, 5):
            if r + 5 <= rows and c + 5 <= cols:
                transform_section(input_grid, output_grid, r, c, section_info)
    
    # Step 4: Handle color replacements
    handle_color_replacements(output_grid)
    
    return output_grid

def analyze_grid(grid: ColoredGrid) -> Dict[Tuple[int, int], Dict[str, any]]:
    rows, cols = grid.get_dimensions()
    section_info = {}
    
    for r in range(0, rows, 5):
        for c in range(0, cols, 5):
            if r + 5 <= rows and c + 5 <= cols:
                section = grid.extract_subgrid(r, c, 5, 5)
                interior = section.extract_subgrid(1, 1, 3, 3)
                colors = [cell for row in interior.values for cell in row]
                color_counts = Counter(colors)
                main_color = max(color_counts, key=color_counts.get)
                special_colors = {color: (i, j) for i, row in enumerate(interior.values) 
                                  for j, color in enumerate(row) if color in [1, 2, 3, 4, 6, 7, 8]}
                
                section_info[(r//5, c//5)] = {
                    'main_color': main_color,
                    'special_colors': special_colors
                }
    
    return section_info

def transform_section(input_grid: ColoredGrid, output_grid: ColoredGrid, r: int, c: int, section_info: Dict[Tuple[int, int], Dict[str, any]]):
    section = input_grid.extract_subgrid(r, c, 5, 5)
    new_section = section.deep_copy()
    
    # Get info for this section
    info = section_info[(r//5, c//5)]
    main_color = info['main_color']
    
    # Apply special colors
    for color, (i, j) in info['special_colors'].items():
        new_section.values[i+1][j+1] = color
    
    # Fill remaining spaces with main color
    for i in range(1, 4):
        for j in range(1, 4):
            if new_section.values[i][j] == 0:
                new_section.values[i][j] = main_color
    
    # Copy to output grid
    for i in range(5):
        for j in range(5):
            output_grid.values[r+i][c+j] = new_section.values[i][j]

def handle_color_replacements(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 6:
                grid.values[r][c] = 1
            elif grid.values[r][c] == 4:
                # Check if there's another special color in the 3x3 section
                for i in range(max(0, r-1), min(rows, r+2)):
                    for j in range(max(0, c-1), min(cols, c+2)):
                        if grid.values[i][j] in [2, 3, 7, 8]:
                            grid.values[r][c] = grid.values[i][j]
                            break
                    if grid.values[r][c] != 4:
                        break
