import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pyvista as pv


data = {
    25: {
        'dt': np.array([1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.175, 1.15, 1.146875, 1.14375, 1.1375, 1.125, 1.1, 1.075, 1.05, 1.0]),
        'stability': np.array([False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True])
    },
    26: {
        'dt': np.array([1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.195, 1.19, 1.18, 1.15]),
        'stability': np.array([False, False, False, False, False, False, False, False, True, True, True])
    },
    27: {
        'dt': np.array([1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2375, 1.225, 1.2]),
        'stability': np.array([False, False, False, False, False, False, True, True, True])
    },
    28: {
        'dt': np.array([1.5, 1.45, 1.4, 1.35, 1.3, 1.295, 1.28, 1.275, 1.25, 1.2]),
        'stability': np.array([False, False, False, False, False, False, True, True, True, True])
    },
    29: {
        'dt': np.array([1.5, 1.45, 1.4, 1.35, 1.33, 1.325, 1.3, 1.295, 1.275, 1.25, 1.2]),
        'stability': np.array([False, False, False, False, False, True, True, True, True, True, True])
    },
    30: {
        'dt': np.array([1.5, 1.45, 1.4, 1.3875, 1.38, 1.375, 1.35, 1.3, 1.25, 1.2]),
        'stability': np.array([False, False, False, False, False, True, True, True, True, True])
    }
}

interface_points = []
for dx in data.keys():
    dt_values = data[dx]['dt']
    stability = data[dx]['stability']
    
    if True in stability:
        first_stable_idx = np.where(stability)[0][0]
        if first_stable_idx > 0:
            interface_dt = (dt_values[first_stable_idx-1] + dt_values[first_stable_idx]) / 2
        else:
            interface_dt = dt_values[first_stable_idx]
        interface_points.append((dx, interface_dt))

interface_x = np.array([point[0] for point in interface_points])
interface_y = np.array([point[1] for point in interface_points])

# Fonction de régression
def power_law(x, a, b):
    return a * (x**b)

# Régression
popt, pcov = curve_fit(power_law, interface_x, interface_y)
a_fit, b_fit = popt

# Fonction théorique
def theoretical_line(x, h_max=21.875):
    c = np.sqrt(9.81 * h_max)
    m = (c)*(np.sqrt(2))
    print(1/m)
    return x / m

# Création de la figure avec deux subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plt.rcParams.update({'font.size': 14})

# Premier subplot : données de stabilité
for dx in data.keys():
    dt_values = data[dx]['dt']
    stability = data[dx]['stability']
    
    ax1.scatter(np.full_like(dt_values[stability], dx), 
                dt_values[stability], 
                c='green', 
                label='Stable' if dx == 25 else "", 
                alpha=0.6)
    
    ax1.scatter(np.full_like(dt_values[~stability], dx), 
                dt_values[~stability], 
                c='red', 
                label='Unstable' if dx == 25 else "", 
                alpha=0.6)

# Deuxième subplot : comparaison des courbes
x_fit = np.linspace(24.5, 30.5, 100)
y_fit = power_law(x_fit, a_fit, b_fit)
y_theo = theoretical_line(x_fit)

# Utiliser des marqueurs différents pour les deux courbes
ax2.plot(x_fit, y_fit, 'b--', marker='o', markevery=10, markersize=8, label=f'Fit: {a_fit:.2e}*x^{b_fit:.3f}')
ax2.plot(x_fit, y_theo, 'k:', marker='s', markevery=10, markersize=8, label='Theoretical CFL')

print("1/a_fit = ", 1/a_fit)

# Personnalisation des subplots
for ax in [ax1, ax2]:
    ax.set_xlabel('dx', fontsize=16)
    ax.set_ylabel('dt', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("Presentation/cfl_plot.svg")
plt.show()



# Afficher les paramètres
print(f"Fit parameters:")
print(f"a = {a_fit:.6e}")
print(f"b = {b_fit:.6f}")

grid_u = pv.read("omp_mpi_u.vti")
nx, ny = grid_u.dimensions[:2]
u = grid_u['u'].reshape((ny, nx))

l2_norm = np.sqrt(np.mean(u**2))

print("u_mean = ", l2_norm)

