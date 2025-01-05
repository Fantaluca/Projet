import numpy as np
import matplotlib.pyplot as plt

# Augmenter la taille de police par défaut
plt.rcParams.update({'font.size': 14})

# Paramètres
t = np.linspace(0, 10, 1000)
omega = 2  # fréquence angulaire
tau = 2    # constante d'amortissement

# Création des fonctions
f_oscillation = np.sin(omega * t)
f_amortie = np.sin(omega * t) * np.exp(-t/tau)

# Création de la figure
plt.figure(figsize=(9, 5))

# Plot des deux fonctions
plt.plot(t, f_oscillation, 'b-', label='$x(t)$', linewidth=2)
plt.plot(t, f_amortie, 'r-', label='$x(t) \\times \\exp{(-t/\\tau)}$', linewidth=2)

plt.xlabel('Time t', fontsize=16)
plt.ylabel('Amplitude(t)', fontsize=16)
plt.legend(fontsize=14)

# Augmenter la taille des ticks sur les axes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig('Presentation/pml_plot.svg')
plt.show()