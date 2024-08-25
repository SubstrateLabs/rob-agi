from rob_agi.colored_grid import ColoredGrid
import numpy as np
from collections import defaultdict
import heapq
from typing import Tuple, List, Dict

def solve_e66aafb8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e66aafb8 challenge by analyzing color transitions and creating a representative pattern.
    
    The function works as follows:
    1. Preprocess the input grid to create a numpy array and identify non-black cells.
    2. Analyze color transitions in the input grid, creating a transition graph.
    3. Identify significant color transitions based on frequency and color contrast.
    4. Determine the output grid size and orientation based on significant transitions.
    5. Construct the output grid by placing colors according to significant transitions.
    6. Post-process the output grid to ensure no black cells and minimum size.
    7. Return the result as a ColoredGrid object.
    
    Returns:
        ColoredGrid: A new grid containing the extracted representative pattern.
    """
    rows, cols = input_grid.get_dimensions()
    grid = np.array([[input_grid.get_cell(r, c) for c in range(cols)] for r in range(rows)])
    non_black_mask = grid != 0
    
    if not np.any(non_black_mask):
        return ColoredGrid(values=[[1, 1], [1, 1]])  # Return a 2x2 blue grid if input is all black
    
    def analyze_transitions(grid: np.ndarray) -> Dict[Tuple[int, int], int]:
        transitions = defaultdict(int)
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    if c < cols - 1 and grid[r, c+1] != 0:
                        transitions[(grid[r, c], grid[r, c+1])] += 1
                    if r < rows - 1 and grid[r+1, c] != 0:
                        transitions[(grid[r, c], grid[r+1, c])] += 1
        return transitions
    
    def calculate_significance(color1: int, color2: int, frequency: int, total: int) -> float:
        return (frequency / total) * (abs(color1 - color2) / 9)
    
    transitions = analyze_transitions(grid)
    total_transitions = sum(transitions.values())
    
    significant_transitions = [
        (-calculate_significance(c1, c2, freq, total_transitions), c1, c2)
        for (c1, c2), freq in transitions.items()
    ]
    heapq.heapify(significant_transitions)
    
    horizontal_transitions = sum(transitions[(c1, c2)] for (c1, c2) in transitions if c1 != c2)
    vertical_transitions = total_transitions - horizontal_transitions
    is_vertical = vertical_transitions > horizontal_transitions
    
    min_size = 2
    max_size = min(8, min(rows, cols) // 3)
    size = min(max(min_size, int(len(significant_transitions) ** 0.5)), max_size)
    
    output_shape = (size, size) if is_vertical else (size, size)
    output_grid = np.zeros(output_shape, dtype=int)
    
    def place_color(grid: np.ndarray, color: int, position: Tuple[int, int]) -> None:
        if 0 <= position[0] < grid.shape[0] and 0 <= position[1] < grid.shape[1]:
            grid[position] = color
    
    def get_available_edges(grid: np.ndarray) -> List[Tuple[int, int]]:
        edges = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                            edges.append((nr, nc))
        return edges
    
    # Place the most significant transition in the center
    _, c1, c2 = heapq.heappop(significant_transitions)
    center = (output_shape[0] // 2, output_shape[1] // 2)
    place_color(output_grid, c1, center)
    place_color(output_grid, c2, (center[0], center[1] + 1))
    
    while significant_transitions and np.any(output_grid == 0):
        edges = get_available_edges(output_grid)
        if not edges:
            break
        
        _, c1, c2 = heapq.heappop(significant_transitions)
        for edge in edges:
            neighbors = [
                output_grid[r, c]
                for r in range(max(0, edge[0] - 1), min(output_shape[0], edge[0] + 2))
                for c in range(max(0, edge[1] - 1), min(output_shape[1], edge[1] + 2))
                if output_grid[r, c] != 0
            ]
            if c1 in neighbors:
                place_color(output_grid, c2, edge)
                break
            elif c2 in neighbors:
                place_color(output_grid, c1, edge)
                break
    
    # Fill any remaining gaps
    for r in range(output_shape[0]):
        for c in range(output_shape[1]):
            if output_grid[r, c] == 0:
                neighbors = [
                    output_grid[nr, nc]
                    for nr in range(max(0, r - 1), min(output_shape[0], r + 2))
                    for nc in range(max(0, c - 1), min(output_shape[1], c + 2))
                    if output_grid[nr, nc] != 0
                ]
                if neighbors:
                    output_grid[r, c] = max(set(neighbors), key=neighbors.count)
                else:
                    output_grid[r, c] = 1  # Default to blue if no neighbors
    
    if is_vertical:
        output_grid = output_grid.T
    
    # Ensure minimum size of 2x2
    if output_grid.shape[0] < 2 or output_grid.shape[1] < 2:
        output_grid = np.pad(output_grid, ((0, max(0, 2 - output_grid.shape[0])), (0, max(0, 2 - output_grid.shape[1]))), mode='edge')
    
    return ColoredGrid(values=output_grid.tolist())
