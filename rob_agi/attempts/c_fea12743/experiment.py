from rob_agi.attempts.c_fea12743.main import get_quadrant

def test_get_quadrant():
    grid_size = 16
    mid_row, mid_col = grid_size // 2, grid_size // 2
    
    for row in range(grid_size):
        for col in range(grid_size):
            quadrant = get_quadrant(row, col, mid_row, mid_col)
            print(f"Row: {row}, Col: {col}, Quadrant: {quadrant}")

    # Test edge cases
    print("\nEdge cases:")
    print(f"Mid-point: {get_quadrant(mid_row, mid_col, mid_row, mid_col)}")
    print(f"Top-left corner: {get_quadrant(0, 0, mid_row, mid_col)}")
    print(f"Top-right corner: {get_quadrant(0, grid_size-1, mid_row, mid_col)}")
    print(f"Bottom-left corner: {get_quadrant(grid_size-1, 0, mid_row, mid_col)}")
    print(f"Bottom-right corner: {get_quadrant(grid_size-1, grid_size-1, mid_row, mid_col)}")

if __name__ == "__main__":
    test_get_quadrant()
