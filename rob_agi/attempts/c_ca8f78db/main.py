from rob_agi.colored_grid import ColoredGrid
import heapq
from collections import Counter
from typing import List, Tuple

def solve_ca8f78db(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Fills in all black (0) cells in the input grid based on the surrounding pattern.
    
    The function uses a priority queue to process black cells, starting with those
    that have the most non-black neighbors. It analyzes the pattern of surrounding
    cells to determine the appropriate color to fill each black cell, maintaining
    the overall pattern of the grid.
    
    Args:
    input_grid (ColoredGrid): The input grid with black cells to be filled.
    
    Returns:
    ColoredGrid: A new grid with all black cells filled according to the pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    black_cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 0]
    
    pq = []
    for r, c in black_cells:
        priority = -count_non_black_neighbors(output_grid, r, c)
        heapq.heappush(pq, (priority, r, c))
    
    while pq:
        _, r, c = heapq.heappop(pq)
        if output_grid.get_cell(r, c) != 0:
            continue
        
        color = determine_fill_color(output_grid, r, c)
        output_grid.set_cell(r, c, color)
        
        for nr, nc in get_neighbors(r, c, rows, cols):
            if output_grid.get_cell(nr, nc) == 0:
                priority = -count_non_black_neighbors(output_grid, nr, nc)
                heapq.heappush(pq, (priority, nr, nc))
    
    # Handle any remaining isolated black cells
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 0:
                color = most_common_color_in_row_or_column(output_grid, r, c)
                output_grid.set_cell(r, c, color)
    
    return output_grid

def count_non_black_neighbors(grid: ColoredGrid, r: int, c: int) -> int:
    return sum(1 for nr, nc in get_neighbors(r, c, *grid.get_dimensions()) if grid.get_cell(nr, nc) != 0)

def get_neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    return [(nr, nc) for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if 0 <= nr < rows and 0 <= nc < cols]

def determine_fill_color(grid: ColoredGrid, r: int, c: int) -> int:
    rows, cols = grid.get_dimensions()
    neighbors = get_neighbors(r, c, rows, cols)
    colors = [grid.get_cell(nr, nc) for nr, nc in neighbors if grid.get_cell(nr, nc) != 0]
    
    if not colors:
        return most_common_color_in_row_or_column(grid, r, c)
    
    if len(set(colors)) == 1:
        return colors[0]
    
    # Check for alternating pattern
    if len(colors) >= 2 and colors[0] != colors[1]:
        return colors[1] if (r + c) % 2 == 0 else colors[0]
    
    # Check for repeating sequence
    sequence = detect_sequence(grid, r, c)
    if sequence:
        return sequence[(r + c) % len(sequence)]
    
    # Default to most common color
    return Counter(colors).most_common(1)[0][0]

def detect_sequence(grid: ColoredGrid, r: int, c: int) -> List[int]:
    rows, cols = grid.get_dimensions()
    row_sequence = [grid.get_cell(r, i) for i in range(cols) if grid.get_cell(r, i) != 0]
    col_sequence = [grid.get_cell(i, c) for i in range(rows) if grid.get_cell(i, c) != 0]
    
    for sequence in [row_sequence, col_sequence]:
        for length in range(2, len(sequence) // 2 + 1):
            if sequence[:length] * (len(sequence) // length) == sequence[:-(len(sequence) % length) or None]:
                return sequence[:length]
    
    return []

def most_common_color_in_row_or_column(grid: ColoredGrid, r: int, c: int) -> int:
    rows, cols = grid.get_dimensions()
    row_colors = [grid.get_cell(r, i) for i in range(cols) if grid.get_cell(r, i) != 0]
    col_colors = [grid.get_cell(i, c) for i in range(rows) if grid.get_cell(i, c) != 0]
    all_colors = row_colors + col_colors
    return Counter(all_colors).most_common(1)[0][0] if all_colors else 1  # Default to blue (1) if no colors found
