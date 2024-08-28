from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_05a7bcf2.main import solve_05a7bcf2, calculate_center_of_mass, find_sky_blue_barrier

def create_test_grid(rows, cols, colored_cells):
    grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r, c, color in colored_cells:
        grid.values[r][c] = color
    return grid

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

# Test case 1: No existing barrier
test_grid_1 = create_test_grid(10, 10, [(2, 2, 4), (7, 7, 2)])
print("Test case 1: No existing barrier")
print("Input grid:")
print_grid(test_grid_1)
center_y, center_x = calculate_center_of_mass(test_grid_1)
orientation, barrier_pos = find_sky_blue_barrier(test_grid_1)
print(f"Center of mass: ({center_y}, {center_x})")
print(f"Barrier: {orientation} at position {barrier_pos}")
result_1 = solve_05a7bcf2(test_grid_1)
print("Output grid:")
print_grid(result_1)

# Test case 2: Existing horizontal barrier
test_grid_2 = create_test_grid(10, 10, [(2, 2, 4), (7, 7, 2)] + [(5, i, 8) for i in range(10)])
print("Test case 2: Existing horizontal barrier")
print("Input grid:")
print_grid(test_grid_2)
center_y, center_x = calculate_center_of_mass(test_grid_2)
orientation, barrier_pos = find_sky_blue_barrier(test_grid_2)
print(f"Center of mass: ({center_y}, {center_x})")
print(f"Barrier: {orientation} at position {barrier_pos}")
result_2 = solve_05a7bcf2(test_grid_2)
print("Output grid:")
print_grid(result_2)

# Test case 3: Existing vertical barrier
test_grid_3 = create_test_grid(10, 10, [(2, 2, 4), (7, 7, 2)] + [(i, 5, 8) for i in range(10)])
print("Test case 3: Existing vertical barrier")
print("Input grid:")
print_grid(test_grid_3)
center_y, center_x = calculate_center_of_mass(test_grid_3)
orientation, barrier_pos = find_sky_blue_barrier(test_grid_3)
print(f"Center of mass: ({center_y}, {center_x})")
print(f"Barrier: {orientation} at position {barrier_pos}")
result_3 = solve_05a7bcf2(test_grid_3)
print("Output grid:")
print_grid(result_3)
