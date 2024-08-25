from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1e81d6f9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving shapes and strategically selecting dots.
    
    1. Preserves the T-shaped gray object.
    2. Counts occurrences of each color.
    3. Identifies connected regions for each color.
    4. Selects up to 3 cells for each color, prioritizing larger shapes and strategic positions.
    5. Performs a final pass to add additional cells if possible without exceeding color limits.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Step 1: Preserve the T-shaped gray object
    preserve_t_shape(input_grid, output_grid)
    
    # Step 2: Count occurrences of each color
    color_counts = count_colors(input_grid)
    
    # Step 3 & 4: Identify connected regions and select strategic dots
    for color in color_counts:
        if color != 5:  # Skip gray
            regions = input_grid.find_connected_regions(color)
            selected_cells = select_strategic_cells(regions, rows, cols, max_cells=3)
            for r, c in selected_cells:
                output_grid.values[r][c] = color
    
    # Step 5: Final pass
    final_pass(input_grid, output_grid, color_counts)
    
    return output_grid

def preserve_t_shape(input_grid: ColoredGrid, output_grid: ColoredGrid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] == 5:  # Gray color
                output_grid.values[r][c] = 5

def count_colors(grid: ColoredGrid) -> Dict[int, int]:
    counts = defaultdict(int)
    for row in grid.values:
        for cell in row:
            if cell != 0 and cell != 5:  # Exclude black and gray
                counts[cell] += 1
    return counts

def select_strategic_cells(regions: List[List[Tuple[int, int]]], rows: int, cols: int, max_cells: int) -> List[Tuple[int, int]]:
    # Sort regions by size, descending
    sorted_regions = sorted(regions, key=len, reverse=True)
    
    selected_cells = []
    for region in sorted_regions:
        if len(selected_cells) >= max_cells:
            break
        
        # Prioritize cells based on their position (corners, edges, etc.)
        priority_cells = []
        for r, c in region:
            priority = 0
            if (r == 0 or r == rows - 1) and (c == 0 or c == cols - 1):  # Corners
                priority = 3
            elif r == 0 or r == rows - 1 or c == 0 or c == cols - 1:  # Edges
                priority = 2
            elif r == rows // 2 and c == cols // 2:  # Center
                priority = 1
            priority_cells.append((priority, (r, c)))
        
        priority_cells.sort(reverse=True)
        selected_cells.extend([cell for _, cell in priority_cells[:max_cells - len(selected_cells)]])
    
    return selected_cells[:max_cells]

def final_pass(input_grid: ColoredGrid, output_grid: ColoredGrid, color_counts: Dict[int, int]):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] != 0 and output_grid.values[r][c] == 0:
                color = input_grid.values[r][c]
                if color != 5 and color_counts[color] < 3:
                    output_grid.values[r][c] = color
                    color_counts[color] += 1
