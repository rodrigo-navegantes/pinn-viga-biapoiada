import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

L       =  1.0    # comprimento da viga [m]
EI_TRUE =  1.0    # valor REAL (desconhecido para a rede)
q       =  1.0    # carga distribuída [N/m]
N_col   =  200    # pontos de colocação (interior)
N_data  =  20     # pontos de medição (sensores)
NOISE   =  0.02   # nível de ruído: 2% do valor máximo de w
EPOCHS  =  50000
LR      =  1e-3

W_PDE  = 1.0
W_BC   = 10.0
W_DATA = 100.0   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

def w_analytical(x_np, EI=EI_TRUE):
    return (q / (24.0 * EI)) * x_np * (L**3 - 2*L*x_np**2 + x_np**3)

# Posições dos sensores 
x_data_np = np.linspace(0.05, 0.95, N_data).astype(np.float32)
w_clean   = w_analytical(x_data_np)

# Adiciona ruído gaussiano
noise_std = NOISE * np.max(np.abs(w_clean))
w_noisy   = w_clean + np.random.normal(0, noise_std, size=w_clean.shape).astype(np.float32)

print(f"\nDados sintéticos gerados:")
print(f"  Sensores     : {N_data} pontos em x ∈ [0.05, 0.95]")
print(f"  Ruído (std)  : {noise_std:.2e}  ({NOISE*100:.0f}% do máx. de w)")
print(f"  w_max exato  : {w_clean.max():.4f}")

x_data = torch.tensor(x_data_np.reshape(-1, 1), device=DEVICE)
w_data = torch.tensor(w_noisy.reshape(-1, 1),   device=DEVICE)

class PINN(nn.Module):
    """
    MLP: 1 → [64x64x64x64] → 1
    Função de ativação: tanh
    """
    def __init__(self, hidden=64, layers=4):
        super().__init__()

        seq = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*seq)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Chute inicial: EI_init = 0.5
        self.log_EI = nn.Parameter(torch.log(torch.tensor(0.5)))

    @property
    def EI(self):
        #Otimizar log(EI) em vez de EI diretamente melhora a estabilidade.
        return torch.exp(self.log_EI)

    def forward(self, x):
        return self.net(x)

# Cálculo do gradiente da função
def nth_derivative(f, x, n):
    df = f
    for _ in range(n):
            df, = torch.autograd.grad(
            df, x,
            grad_outputs=torch.ones_like(df),
            create_graph=True,
            retain_graph=True,
        )
    return df

def compute_loss(model, x_col):
    EI_pred = model.EI  

    # L_pde / Resíduo da EDP
    w_col   = model(x_col)
    d4w     = nth_derivative(w_col, x_col, 4)
    loss_pde = torch.mean((EI_pred * d4w - q) ** 2)

    # L_bcc / Condições de contorno
    x0 = torch.tensor([[0.0]], device=DEVICE, requires_grad=True)
    xL = torch.tensor([[L]],   device=DEVICE, requires_grad=True)

    w0       = model(x0)
    wL       = model(xL)
    d2w_x0   = nth_derivative(model(x0), x0, 2)
    d2w_xL   = nth_derivative(model(xL), xL, 2)

    loss_bc = (w0**2 + wL**2 + d2w_x0**2 + d2w_xL**2).mean()

    # L_data / Erro nos dados medidos
    w_pred_data = model(x_data)
    loss_data   = torch.mean((w_pred_data - w_data) ** 2)

    # L_total  Perda total ponderada 
    loss_total = W_PDE * loss_pde + W_BC * loss_bc + W_DATA * loss_data

    return loss_total, {
        "pde":  loss_pde.item(),
        "bc":   loss_bc.item(),
        "data": loss_data.item(),
        "EI":   EI_pred.item(),
    }

model     = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=500, factor=0.5, min_lr=1e-5
)

x_col_np = np.linspace(0, L, N_col).reshape(-1, 1).astype(np.float32)
x_col    = torch.tensor(x_col_np, device=DEVICE, requires_grad=True)

history = {"total": [], "pde": [], "bc": [], "data": [], "EI": []}

print(f"\n{'Época':>8}  {'Loss':>10}  {'L_pde':>10}  {'L_bc':>8}  {'L_data':>10}  {'EI_pred':>10}  {'Erro EI':>8}")
print("─" * 80)

for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    loss, parts = compute_loss(model, x_col)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    history["total"].append(loss.item())
    history["pde"].append(parts["pde"])
    history["bc"].append(parts["bc"])
    history["data"].append(parts["data"])
    history["EI"].append(parts["EI"])

    # Printar para monitorar a convergência a cada 2000 épocas
    if epoch % 2000 == 0 or epoch == 1:
        erro_EI = abs(parts["EI"] - EI_TRUE) / EI_TRUE * 100
        print(f"{epoch:>8}  {loss.item():>10.4e}  {parts['pde']:>10.4e}  "
              f"{parts['bc']:>8.4e}  {parts['data']:>10.4e}  "
              f"{parts['EI']:>10.5f}  {erro_EI:>7.3f}%")

EI_final = model.EI.item()
erro_final = abs(EI_final - EI_TRUE) / EI_TRUE * 100
print(f"\n{'─'*50}")
print(f"  EI real       : {EI_TRUE:.5f}")
print(f"  EI inicial    : 0.5)")
print(f"  EI identificado: {EI_final:.5f}")
print(f"  Erro relativo  : {erro_final:.3f}%")
print(f"{'─'*50}")

# Avaliação final do modelo
model.eval()
x_test_np = np.linspace(0, L, 300, dtype=np.float32).reshape(-1, 1)
x_test    = torch.tensor(x_test_np, device=DEVICE)

with torch.no_grad(): w_pred_np = model(x_test).cpu().numpy().flatten()

w_exact_np = w_analytical(x_test_np.flatten())
rel_error  = np.linalg.norm(w_pred_np - w_exact_np) / np.linalg.norm(w_exact_np)
print(f"  Erro L² (w)    : {rel_error:.4e}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"PINN - Problema Inverso: Identificação de EI ({EPOCHS} ITERAÇÕES)", fontsize=13, fontweight="bold")

# ── Gráfico 1: Deslocamento ────────────────────────────────
ax = axes[0]
ax.plot(x_test_np,    w_exact_np * 1e3, "k-",  lw=2.5, label="Analítica (EI real)")
ax.plot(x_test_np,    w_pred_np  * 1e3, "r--", lw=2,   label=f"PINN (EI={EI_final:.4f})")
ax.scatter(x_data_np, w_noisy    * 1e3, s=30, color="steelblue",
           zorder=5, label=f"Dados medidos ({N_data} pts, ruído {NOISE*100:.0f}%)")
ax.set_xlabel("x [m]")
ax.set_ylabel("w [mm]")
ax.set_title("Deslocamento transversal")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.35)

# ── Gráfico 2: Convergência de EI ─────────────────────────
ax = axes[1]
epochs_arr = np.arange(1, EPOCHS + 1)
ax.axhline(EI_TRUE, color="k",  lw=2,   linestyle="-",  label=f"EI real = {EI_TRUE}")
ax.axhline(0.5,     color="gray", lw=1, linestyle=":",  label="EI inicial = 0.5")
ax.plot(epochs_arr, history["EI"], color="crimson", lw=1.5, label="EI identificado")
ax.set_xlabel("Época")
ax.set_ylabel("EI")
ax.set_title(f"Convergência de EI  (erro final = {erro_final:.3f}%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35)

# ── Gráfico 3: Histórico de perda ─────────────────────────
ax = axes[2]
ax.semilogy(history["total"], "k-",  lw=1.5, label="Total")
ax.semilogy(history["pde"],   "b--", lw=1.2, label="L_pde")
ax.semilogy(history["bc"],    "g-.", lw=1.2, label="L_bc")
ax.semilogy(history["data"],  "r:",  lw=1.5, label="L_data")
ax.set_xlabel("Época")
ax.set_ylabel("MSE")
ax.set_title("Histórico de treinamento")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.35)

plt.tight_layout()

import os
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pinn_inverso_EI.png", dpi=150, bbox_inches="tight")
plt.show()
print(" \nGráfico salvo em outputs/pinn_inverso_EI.png")
