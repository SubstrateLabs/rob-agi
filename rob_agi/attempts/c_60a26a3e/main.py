from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_60a26a3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red shapes with minimal blue lines.
    
    The solution follows these steps:
    1. Identify all red (2) shapes
    2. Calculate the center of each red shape
    3. Create a minimal spanning tree (MST) connecting all shape centers
    4. Optimize vertical connections
    5. Create blue connections based on the optimized MST
    6. Optimize connections by removing unnecessary blue cells
    7. Fill in small gaps to ensure continuous connections
    8. Validate the solution to ensure all red shapes are connected
    
    This approach creates a minimal structure of blue lines that efficiently connects
    red shapes, focusing on creating a skeletal structure rather than filling all gaps.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify red shapes
    red_shapes = find_red_shapes(input_grid)
    
    if not red_shapes:
        return output_grid
    
    # Step 2: Calculate shape centers
    shape_centers = [calculate_shape_center(shape) for shape in red_shapes]
    
    # Step 3: Create minimal spanning tree
    mst = create_minimal_spanning_tree(shape_centers)
    
    # Step 4: Optimize vertical connections
    optimized_mst = optimize_vertical_connections(mst, input_grid)
    
    # Step 5: Create blue connections
    create_blue_connections(output_grid, optimized_mst, red_shapes)
    
    # Step 6: Optimize connections
    optimize_connections(output_grid)
    
    # Step 7: Fill in gaps
    fill_gaps(output_grid)
    
    # Step 8: Validate solution
    if not validate_solution(output_grid):
        raise ValueError("Invalid solution: not all red shapes are connected")
    
    return output_grid

def find_red_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    shapes = []
    
    def dfs(r, c):
        shape = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == 2:
                visited.add((curr_r, curr_c))
                shape.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return shape
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 2 and (r, c) not in visited:
                shapes.append(dfs(r, c))
    
    return shapes

def calculate_shape_center(shape: List[Tuple[int, int]]) -> Tuple[int, int]:
    avg_r = sum(r for r, _ in shape) / len(shape)
    avg_c = sum(c for _, c in shape) / len(shape)
    return (round(avg_r), round(avg_c))

def create_minimal_spanning_tree(centers: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    edges = []
    for i, (r1, c1) in enumerate(centers):
        for j, (r2, c2) in enumerate(centers[i+1:], i+1):
            distance = abs(r1 - r2) + abs(c1 - c2)
            edges.append((distance, i, j, r1, c1, r2, c2))
    
    edges.sort()
    parent = list(range(len(centers)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    mst = []
    for _, i, j, r1, c1, r2, c2 in edges:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            mst.append((r1, c1, r2, c2))
    
    return mst

def optimize_vertical_connections(mst: List[Tuple[int, int, int, int]], grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    rows, cols = grid.get_dimensions()
    optimized_mst = []
    
    for r1, c1, r2, c2 in mst:
        if r1 == r2:  # Horizontal connection
            optimized_mst.append((r1, c1, r2, c2))
        else:  # Vertical or diagonal connection
            best_col = c1
            min_blue_cells = float('inf')
            
            for test_col in range(max(0, min(c1, c2) - 2), min(cols, max(c1, c2) + 3)):
                blue_cells = sum(1 for r in range(min(r1, r2), max(r1, r2) + 1) if grid.values[r][test_col] == 0)
                if blue_cells < min_blue_cells:
                    min_blue_cells = blue_cells
                    best_col = test_col
            
            optimized_mst.append((r1, c1, r1, best_col))
            optimized_mst.append((r1, best_col, r2, best_col))
            optimized_mst.append((r2, best_col, r2, c2))
    
    return optimized_mst

def create_blue_connections(grid: ColoredGrid, mst: List[Tuple[int, int, int, int]], red_shapes: List[List[Tuple[int, int]]]):
    for r1, c1, r2, c2 in mst:
        if r1 == r2:  # Horizontal connection
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if grid.values[r1][c] == 0:
                    grid.values[r1][c] = 1
        else:  # Vertical connection
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if grid.values[r][c1] == 0:
                    grid.values[r][c1] = 1

def optimize_connections(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                adjacent_red = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                   if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.values[r + dr][c + dc] == 2)
                if adjacent_red > 1:
                    grid.values[r][c] = 0

def fill_gaps(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                adjacent_blue_or_red = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                           if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.values[r + dr][c + dc] in [1, 2])
                if adjacent_blue_or_red > 2:
                    grid.values[r][c] = 1

def validate_solution(grid: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    visited = set()
    
    def dfs(r, c):
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] in [1, 2]:
                visited.add((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
    
    # Start DFS from the first red cell
    start = next((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 2)
    dfs(*start)
    
    # Check if all red and blue cells are visited
    return all((r, c) in visited for r in range(rows) for c in range(cols) if grid.values[r][c] in [1, 2])
