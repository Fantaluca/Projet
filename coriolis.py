import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

def plot_eta_comparison():
    # Lecture des fichiers pour eta
    grid_with = pv.read("pml_eta_with_coriolis.vti")
    grid_without = pv.read("pml_eta_without_coriolis.vti")
    
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
    plt.savefig('Presentation/coriolis_eta_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_vorticity(u, v, dx, dy):
    """Calcule la vorticité à partir des composantes u et v de la vitesse"""
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    return dvdx - dudy

def calculate_rotation_vectors(vorticity):
    """Calcule les vecteurs de rotation à partir de la vorticité"""
    # Créer des vecteurs qui tournent autour des points de vorticité
    # Le sens de rotation dépend du signe de la vorticité
    u_rot = -np.gradient(vorticity, axis=0)  # composante x de la rotation
    v_rot = np.gradient(vorticity, axis=1)   # composante y de la rotation
    return u_rot, v_rot

def plot_vorticity_comparison():
    # Lecture des fichiers
    grid_u_with = pv.read("pml_u_with_coriolis.vti")
    grid_v_with = pv.read("pml_v_with_coriolis.vti")
    
    grid_u_without = pv.read("pml_u_without_coriolis.vti")
    grid_v_without = pv.read("pml_v_without_coriolis.vti")
    
    # Extraction des composantes de vitesse
    u_with = grid_u_with['u']
    v_with = grid_v_with['v']
    u_without = grid_u_without['u']
    v_without = grid_v_without['v']
    
    # Dimensions et grille
    nx, ny = grid_u_with.dimensions[:2]
    x = np.linspace(grid_u_with.bounds[0], grid_u_with.bounds[1], nx)
    y = np.linspace(grid_u_with.bounds[2], grid_u_with.bounds[3], ny)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Reshape des données
    u_with_2d = u_with.reshape((ny, nx))
    v_with_2d = v_with.reshape((ny, nx))
    u_without_2d = u_without.reshape((ny, nx))
    v_without_2d = v_without.reshape((ny, nx))
    
    # Calcul des vorticités
    vorticity_with = calculate_vorticity(u_with_2d, v_with_2d, dx, dy)
    vorticity_without = calculate_vorticity(u_without_2d, v_without_2d, dx, dy)
    
    # Calcul des vecteurs de rotation
    u_rot_with, v_rot_with = calculate_rotation_vectors(vorticity_with)
    u_rot_without, v_rot_without = calculate_rotation_vectors(vorticity_without)
    
    # Création de la grille pour les flèches
    skip = 8
    X, Y = np.meshgrid(x[::skip], y[::skip])
    
    # Masques pour la vorticité positive
    mask_with = vorticity_with[::skip, ::skip] > 0
    mask_without = vorticity_without[::skip, ::skip] > 0
    
    # Application des masques aux vecteurs de rotation
    U_rot_with = np.where(mask_with, u_rot_with[::skip, ::skip], 0)
    V_rot_with = np.where(mask_with, v_rot_with[::skip, ::skip], 0)
    U_rot_without = np.where(mask_without, u_rot_without[::skip, ::skip], 0)
    V_rot_without = np.where(mask_without, v_rot_without[::skip, ::skip], 0)
    
    # Normalisation seulement où la vorticité est positive
    norm_with = np.sqrt(U_rot_with**2 + V_rot_with**2)
    mask_nonzero_with = norm_with > 0
    U_rot_with[mask_nonzero_with] /= norm_with[mask_nonzero_with]
    V_rot_with[mask_nonzero_with] /= norm_with[mask_nonzero_with]
    
    norm_without = np.sqrt(U_rot_without**2 + V_rot_without**2)
    mask_nonzero_without = norm_without > 0
    U_rot_without[mask_nonzero_without] /= norm_without[mask_nonzero_without]
    V_rot_without[mask_nonzero_without] /= norm_without[mask_nonzero_without]
    
    # Définition des limites de l'échelle de couleur
    vmax = max(abs(vorticity_with).max(), abs(vorticity_without).max())
    vmin = -vmax
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Vorticité sans Coriolis
    im1 = ax1.pcolormesh(x, y, vorticity_without, shading='nearest', 
                        cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.quiver(X[mask_without], Y[mask_without], 
               U_rot_without[mask_without], V_rot_without[mask_without],
               color='black', alpha=0.6, scale=15, width=0.003)
    plt.colorbar(im1, ax=ax1, label='Vorticity')
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Position y', fontsize=12)
    ax1.set_title('Vorticity field without Coriolis', fontsize=14)
    
    # Vorticité avec Coriolis
    im2 = ax2.pcolormesh(x, y, vorticity_with, shading='nearest', 
                        cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax2.quiver(X[mask_with], Y[mask_with], 
               U_rot_with[mask_with], V_rot_with[mask_with],
               color='black', alpha=0.6, scale=15, width=0.003)
    plt.colorbar(im2, ax=ax2, label='Vorticity')
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('Position y', fontsize=12)
    ax2.set_title('Vorticity field with Coriolis', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('Presentation/vorticity_comparison_with_rotation_positive.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_vorticity_comparison()