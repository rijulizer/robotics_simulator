import pygame
import numpy as np

# Initialize Pygame
# pygame.init()

# Set up the screen
screen_width = 800
screen_height = 600
# screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Ellipse from Covariance Matrix")

# Covariance matrix
cov_matrix = np.array([[2, -1],
                       [1, 2]])
print("#" * 100)
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"eigenvalues, eigenvectors: {eigenvalues, eigenvectors}")
# Find major and minor axes lengths
major_axis = 200 * np.sqrt(eigenvalues[0])
minor_axis = 200 * np.sqrt(eigenvalues[1])
print(f"major_axis, minor_axis: {major_axis, minor_axis}")
# Find angle of rotation
rotation_angle = np.arctan2(float(eigenvectors[1, 0]), float(eigenvectors[0, 0]))
# Center of the ellipse
center = (screen_width // 2, screen_height // 2)

# Rotation matrix
rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle), np.cos(rotation_angle)]])
print(f"rotation_angle, rotation_matrix: {rotation_angle, rotation_matrix}")

# Rotate the major and minor axes
rotated_major_axis, rotated_minor_axis = np.dot(rotation_matrix, [major_axis / 2, minor_axis / 2])
print(f"rotated_major_axis, rotated_minor_axis: {rotated_major_axis, rotated_minor_axis}")
# print((center[0] - rotated_major_axis, center[1] - rotated_minor_axis, 2 * rotated_major_axis, 2 * rotated_minor_axis))
# # Draw the rotated ellipse
# pygame.draw.ellipse(screen, (255, 0, 0), (center[0] - rotated_major_axis, center[1] - rotated_minor_axis, 2 * rotated_major_axis, 2 * rotated_minor_axis), 2)

# # Main loop
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
    
#     pygame.display.flip()

# # Quit Pygame
# pygame.quit()
