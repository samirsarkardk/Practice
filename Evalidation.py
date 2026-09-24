import numpy as np
import torch
from Aconfig import (Config, DEVICE)
from Bmodel import (PINN1, PINN2, PINN3)
import os
import matplotlib.pyplot as plt

N = 5000
t_value = torch.tensor(0.25)

config = Config()

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)


model1.load_state_dict(torch.load("model1.pth", map_location=DEVICE))
model2.load_state_dict(torch.load("model2.pth", map_location=DEVICE))
model3.load_state_dict(torch.load("model3.pth", map_location=DEVICE))

model1.eval()
model2.eval()
model3.eval()

x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.t_max
x_min.to(DEVICE), x_max.to(DEVICE), t_min.to(DEVICE), t_max.to(DEVICE)

t_mid = (t_max + t_min)/2


x = x_min + (x_max - x_min) * torch.rand(N,1)

t = t_min + (t_max - t_min) * torch.rand(N,1)

t_mid.to(DEVICE), x.to(DEVICE), t.to(DEVICE), t_value.to(DEVICE)

def final_solution(x, t, model1, model2, model3, t_mid):
    x = x.reshape(-1, 1).to(DEVICE)
    t = t.reshape(-1, 1).to(DEVICE)
    t_mid = torch.as_tensor(t_mid, device=DEVICE)

    interface_mask = torch.isclose(t, t_mid, atol=1e-11)
    mask1 = (t < t_mid) & (~interface_mask)
    mask2 = (t > t_mid) & (~interface_mask)

    with torch.no_grad():
        u1 = model1(x, t)
        u2 = model2(x, t)
        u3 = model3(x, t)

    u_interface = (u1 + u2 + u3) / 3

    u_final = (
        u1 * mask1.float()
        + u2 * mask2.float()
        + u_interface * interface_mask.float()
    )

    return u_final

def exact_solution(x, t):
    return torch.exp(-torch.pi**2 * t) * torch.sin(torch.pi * x)


# ==========================================================
# Relative L2 Error at t = 0.25
# ==========================================================

N1 = 5000


# Create 5000 x-points from x_min to x_max
x_test = torch.linspace(x_min, x_max, N1).reshape(-1, 1).to(DEVICE)

# Create a t-column where every value is 0.25
t_test = torch.full_like(x_test, t_value).to(DEVICE)

# PINN prediction
u_pred1 = final_solution(
    x_test,
    t_test,
    model1,
    model2,
    model3,
    t_mid
)

# Exact solution at the same x and t points
u_exact1 = exact_solution(x_test, t_test)   # Replace with your exact-solution function

# Relative L2 error
relative_l2_error = (
    torch.linalg.vector_norm(u_pred1 - u_exact1)
    / torch.linalg.vector_norm(u_exact1)
)

print(f"\nRelative L2 Error at t = {t_value}: {relative_l2_error.item():.6e}")
print(f"Relative L2 Error Percentage: {relative_l2_error.item() * 100:.4f}%")


u_pred = final_solution(
    x, t, model1, model2, model3, t_mid)

u_exact = exact_solution(x, t)

np.save("predicted_solution.npy", u_pred.detach().cpu().numpy())


Nx, Nt = 200, 200

x_values = np.linspace(x_min.item(), x_max.item(), Nx)
t_values = np.linspace(torch.tensor(0.25).item(), torch.tensor(0.2500001).item(), Nt)
X, T = np.meshgrid(x_values, t_values)

x_grid = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
t_grid = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

u_pred_grid = final_solution(
    x_grid, t_grid, model1, model2, model3, t_mid
).cpu().numpy().reshape(Nt, Nx)

u_exact_grid = np.exp(-np.pi**2 * T) * np.sin(np.pi * X)
absolute_error = np.abs(u_pred_grid - u_exact_grid)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

vmin = min(u_exact_grid.min(), u_pred_grid.min())
vmax = max(u_exact_grid.max(), u_pred_grid.max())

im0 = ax[0].pcolormesh(X, T, u_exact_grid, shading="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
ax[0].set_title("Exact solution")
ax[0].set_xlabel("x")
ax[0].set_ylabel("t")
fig.colorbar(im0, ax=ax[0])

im1 = ax[1].pcolormesh(X, T, u_pred_grid, shading="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
ax[1].set_title("Predicted solution")
ax[1].set_xlabel("x")
ax[1].set_ylabel("t")
fig.colorbar(im1, ax=ax[1])

im2 = ax[2].pcolormesh(X, T, absolute_error, shading="auto", cmap="magma")
ax[2].set_title("Absolute error")
ax[2].set_xlabel("x")
ax[2].set_ylabel("t")
fig.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.savefig("solution_comparison.png", dpi=300)
plt.show()