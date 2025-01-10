import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

def plot_eta_comparison():
    # Lecture des fichiers pour eta
    grid_with = pv.read("data/pml_eta_with_coriolis.vti")
    grid_without = pv.read("data/pml_eta_without_coriolis.vti")
    
    # Extraction des élévations d'eau
    eta_with = grid_with['water elevation']
    eta_without = grid_without['water elevation']
    
    nx, ny = grid_with.dimensions[:2]
    y_mid = ny // 2
    x = np.linspace(grid_with.bounds[0], grid_with.bounds[1], nx)
    
    # Reshape pour le plotting
    eta_with_2d = eta_with.reshape((ny, nx))
    eta_without_2d = eta_without.reshape((ny, nx))
    
    profile_with = eta_with_2d[y_mid, :]
    profile_without = eta_without_2d[y_mid, :]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, profile_with, 'b-', label='With Coriolis', linewidth=2)
    plt.plot(x, profile_without, 'r--', label='Without Coriolis', linewidth=2)
    
    plt.xlabel('Position x', fontsize=14)
    plt.ylabel('Water elevation η', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../Presentation/coriolis_eta_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_eta_comparison()