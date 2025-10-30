import numpy as np
import matplotlib.pyplot as plt

def mixed_kernel(e, alpha, sigma_g, sigma_l):
    return alpha * np.exp(-(e**2) / sigma_g) + (1 - alpha) * np.exp(-np.abs(e) / sigma_l)

def mixed_kernel_grad(e, alpha, sigma_g, sigma_l):
    return alpha * np.exp(-(e**2) / sigma_g) * (-2*e/sigma_g) + (1-alpha) * np.exp(-e/sigma_l) * (-1/sigma_l)

error = np.linspace(0, 0.12, 400)
e_target = 0.12

# pick candidates
cands_plot = [
    (0.5, 0.0035, 0.010),
    (0.5, 0.0035, 0.012),
    (0.5, 0.0035, 0.014),
    (0.5, 0.0035, 0.010),
    (0.20, 0.0040, 0.011),
]

plt.figure(figsize=(9, 6))
for a, sg, sl in cands_plot:
    r = mixed_kernel(error, a, sg, sl)
    r_end = mixed_kernel(e_target, a, sg, sl)
    g_end = mixed_kernel_grad(e_target, a, sg, sl)
    lbl = f"α={a:.3f}, σg={sg:.4f}, σl={sl:.3f}  (R@0.12={r_end:.4f}, |dR/dx|@0.12={abs(g_end):.3f})"
    plt.plot(error, r, label=lbl)

plt.title("Mixed Kernel — drive R(0.12) → 0 with usable gradient")
plt.xlabel("Error (m)")
plt.ylabel("Reward")
plt.grid(True)
plt.legend(fontsize=9)
plt.show()
