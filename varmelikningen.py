import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parametre
L = np.pi   # Lengden av stangen (0 til pi)
T = 2.0     # Total tid
Nx = 50     # Antall romlige punkter
Nt = 1000   # Antall tidssteg

h = L / Nx
k = T / Nt
alpha = k / h**2  # Stabilitetsbetingelse

# Diskretisert rom og tid
x = np.linspace(0, L, Nx+1)
t = np.linspace(0, T, Nt+1)

# Initialbetingelse
u_explicit = np.sin(x)
u_implicit = np.sin(x)
u_cn = np.sin(x)

# Matrise for implisitt og Crank-Nicolson
A_impl_diag_main = (1 + 2 * alpha) * np.ones(Nx-1)
A_impl_diag_off = -alpha * np.ones(Nx-2)
A_impl = diags([A_impl_diag_off, A_impl_diag_main, A_impl_diag_off], [-1, 0, 1], format="csc")

A_cn_main = (1 + alpha) * np.ones(Nx-1)
A_cn_off = -alpha / 2 * np.ones(Nx-2)
A_cn = diags([A_cn_off, A_cn_main, A_cn_off], [-1, 0, 1], format="csc")
B_cn_main = (1 - alpha) * np.ones(Nx-1)
B_cn_off = alpha / 2 * np.ones(Nx-2)
B_cn = diags([B_cn_off, B_cn_main, B_cn_off], [-1, 0, 1], format="csc")

# Lagring av resultater
u_explicit_all = [u_explicit.copy()]
u_implicit_all = [u_implicit.copy()]
u_cn_all = [u_cn.copy()]
u_analytical_all = [np.exp(-t[0]) * np.sin(x)]

# Løsning
for n in range(Nt):
    # Eksplisitt
    u_exp_new = u_explicit.copy()
    for i in range(1, Nx):
        u_exp_new[i] = u_explicit[i] + alpha * (u_explicit[i+1] - 2*u_explicit[i] + u_explicit[i-1])
    u_explicit = u_exp_new
    u_explicit_all.append(u_explicit.copy())
    
    # Implisitt
    b_impl = u_implicit[1:Nx]
    u_implicit[1:Nx] = spsolve(A_impl, b_impl)
    u_implicit_all.append(u_implicit.copy())
    
    # Crank-Nicolson
    b_cn = B_cn @ u_cn[1:Nx]
    u_cn[1:Nx] = spsolve(A_cn, b_cn)
    u_cn_all.append(u_cn.copy())

    # Analytisk løsning
    u_analytical_all.append(np.exp(-t[n+1]) * np.sin(x))

# Animasjon med 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

lines = []
labels = ["Eksplisitt", "Implisitt", "Crank-Nicolson", "Analytisk"]
data_sets = [u_explicit_all, u_implicit_all, u_cn_all, u_analytical_all]

for ax, label, data in zip(axes, labels, data_sets):
    line, = ax.plot(x, data[0], label=label)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(label)
    ax.legend()
    lines.append(line)

def update(frame):
    for line, data in zip(lines, data_sets):
        line.set_ydata(data[frame])
    return lines

ani = animation.FuncAnimation(fig, update, frames=range(0, Nt, Nt//100), interval=50)
plt.tight_layout()
plt.show()
