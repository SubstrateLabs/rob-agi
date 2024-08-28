from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_15113be4.main import solve_15113be4

def create_test_grid() -> ColoredGrid:
    return ColoredGrid(values=[
        [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 1, 4, 1, 0, 1, 4, 0, 1, 0, 4, 0, 0, 0],
        [4, 8, 8, 0, 0, 8, 8, 4, 0, 0, 1, 4, 0, 1, 0, 4, 0, 0, 0, 4, 0, 1, 0],
        [4, 8, 8, 0, 0, 8, 8, 4, 0, 1, 1, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
        [4, 0, 0, 8, 8, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 0, 0, 8, 8, 0, 0, 4, 1, 1, 1, 4, 0, 1, 0, 4, 0, 0, 0, 4, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ] * 3)  # Repeat the pattern 3 times to create a 23x23 grid

def main():
    input_grid = create_test_grid()
    output_grid = solve_15113be4(input_grid)
    
    print("Input Grid:")
    print(input_grid)
    print("\nOutput Grid:")
    print(output_grid)
    
    # Check if yellow (4) structure is preserved
    yellow_preserved = all(
        input_grid.get_cell(r, c) == 4 and output_grid.get_cell(r, c) == 4
        for r in range(23) for c in range(23)
        if input_grid.get_cell(r, c) == 4
    )
    print(f"\nYellow structure preserved: {yellow_preserved}")
    
    # Count L-shapes
    l_shapes = count_l_shapes(output_grid)
    print(f"Number of L-shapes: {l_shapes}")

def count_l_shapes(grid: ColoredGrid) -> int:
    count = 0
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if is_l_shape(grid, r, c):
                count += 1
    return count

def is_l_shape(grid: ColoredGrid, r: int, c: int) -> bool:
    color = grid.get_cell(r, c)
    if color in [0, 1, 4]:  # Not a secondary color
        return False
    
    patterns = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)]
    ]
    
    return any(
        all(grid.get_cell(r + dr, c + dc) == color for dr, dc in pattern)
        for pattern in patterns
    )

if __name__ == "__main__":
    main()
