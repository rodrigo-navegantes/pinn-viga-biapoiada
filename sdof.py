"""
Estratégia:
    ENTRADA  I = [x(tᵢ); ẍ(tᵢ)],  i = 1..N  (período de treinamento)
    SAÍDA    O = [x(tᵢ); ẍ(tᵢ); ω; ζ],  i > N  (predição + identificação)

Função de perda:
    L = (1/N) Σ [(x*ᵢ - x̂ᵢ)² + (ẍ*ᵢ - ẍ̂ᵢ)²]         ← ajuste aos dados
      + λ·(1/N) Σ [ẍ̂ᵢ + 2ζ̂ω̂·ẋ̂ᵢ + ω̂²·x̂ᵢ]²            ← resíduo físico
      + λ·[x₀* - x̂(0)]²                             ← condição inicial
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# Parâmetros desconhecidos pela a rede
OMEGA_TRUE = 2 * np.pi * 2.0                   # frequência natural [rad/s] → 2 Hz
ZETA_TRUE  = 0.05                              # razão de amortecimento (5%)

# Condições iniciais
X0    = 1.0                                    # deslocamento inicial [m]
XDOT0 = 0.0                                    # velocidade inicial [m/s]

# Domínio temporal
T_TOTAL   = 5.0                                # tempo total simulado [s]
T_TRAIN   = 2.0                                # tempo de treinamento (dados disponíveis) [s]
N_STEPS   = 1000                               # pontos totais
N_TRAIN   = int(N_STEPS * T_TRAIN / T_TOTAL)   # pontos de treinamento

# Hiperparâmetros
EPOCHS    = 20_000
LR        = 1e-3
LAMBDA    = 1.0                                # peso da física (λ=1 → PINN, λ=0 → ANN pura)
NOISE     = 0.02                               # ruído nos dados de treinamento (2%)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo : {DEVICE}")
print(f"ω real      : {OMEGA_TRUE:.4f} rad/s  ({OMEGA_TRUE/(2*np.pi):.2f} Hz)")
print(f"ζ real      : {ZETA_TRUE:.4f}")
print(f"T_train     : {T_TRAIN}s de {T_TOTAL}s  ({N_TRAIN}/{N_STEPS} pontos)")

# Solução Analítca
def sdof_analytical(t, omega, zeta, x0, xdot0):
    omega_d = omega * np.sqrt(1 - zeta**2)
    A = x0
    B = (xdot0 + zeta * omega * x0) / omega_d
    x    = np.exp(-zeta * omega * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    xdot = np.gradient(x, t)                         
    xddot = np.gradient(xdot, t)                     
    return x, xdot, xddot

t_all   = np.linspace(0, T_TOTAL, N_STEPS, dtype=np.float32)
x_clean, xdot_clean, xddot_clean = sdof_analytical(
    t_all, OMEGA_TRUE, ZETA_TRUE, X0, XDOT0
)

# Adiciona ruído
noise_std = NOISE * np.max(np.abs(x_clean))
x_noisy     = x_clean     + np.random.normal(0, noise_std,     N_STEPS).astype(np.float32)
xddot_noisy = xddot_clean + np.random.normal(0, noise_std * 10, N_STEPS).astype(np.float32)

# Separação treino / predição
t_train     = t_all[:N_TRAIN]
x_train     = x_noisy[:N_TRAIN]
xddot_train = xddot_noisy[:N_TRAIN]

print(f"\nDados gerados:")
print(f"  Ruído (std) : {noise_std:.4e}")
print(f"  x_max       : {x_clean.max():.4f}")

class PINN_SDOF(nn.Module):
    def __init__(self, hidden=50, n_layers=1):
        super().__init__()

        #1 camada oculta com 50 nós
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # ω_init = 0.5 * ω_real,  ζ_init = 0.5 (muito maior que 0.05)
        omega_init = 0.5 * OMEGA_TRUE
        zeta_init  = 0.5

        self.raw_omega = nn.Parameter(torch.tensor(
            np.log(np.exp(omega_init) - 1), dtype=torch.float32
        ))
        self.raw_zeta = nn.Parameter(torch.tensor(
            np.log(zeta_init / (1 - zeta_init)), dtype=torch.float32
        ))

    @property
    def omega(self):
        return torch.nn.functional.softplus(self.raw_omega)

    @property
    def zeta(self):
        return torch.sigmoid(self.raw_zeta)

    def forward(self, t):
        return self.net(t)

def time_derivative(f, t, order=1):
    df = f
    for _ in range(order):
        df, = torch.autograd.grad(
            df, t,
            grad_outputs=torch.ones_like(df),
            create_graph=True,
            retain_graph=True,
        )
    return df

# Função de perda
def compute_loss(model, t_data, x_star, xddot_star, t_phys, lam=1.0):
    """
    L = (1/N) Σ [(x* - x̂)² + (ẍ* - ẍ̂)²]         → L_data
      + λ · (1/N) Σ [ẍ̂ + 2ζ̂ω̂ẋ̂ + ω̂²x̂]²           → L_phys 
      + λ · [x₀* - x̂(0)]²                       → L_ic   
    """
    omega = model.omega
    zeta  = model.zeta

    # L_data
    x_pred_data    = model(t_data)
    xdot_pred_data = time_derivative(x_pred_data, t_data, order=1)
    xddot_pred_data = time_derivative(x_pred_data, t_data, order=2)

    loss_x     = torch.mean((x_pred_data    - x_star)  ** 2)
    loss_xddot = torch.mean((xddot_pred_data - xddot_star) ** 2)
    loss_data  = loss_x + loss_xddot

    # L_phys
    x_phys    = model(t_phys)
    xdot_phys = time_derivative(x_phys, t_phys, order=1)
    xddot_phys = time_derivative(x_phys, t_phys, order=2)

    residual  = xddot_phys + 2 * zeta * omega * xdot_phys + omega**2 * x_phys
    loss_phys = torch.mean(residual ** 2)

    # L_ic
    t0     = torch.tensor([[0.0]], device=DEVICE, requires_grad=True)
    x_at_0 = model(t0)
    loss_ic = (x_at_0 - X0) ** 2

    #L_total
    loss_total = loss_data + lam * (loss_phys + loss_ic)

    return loss_total, {
        "data":  loss_data.item(),
        "phys":  loss_phys.item(),
        "ic":    loss_ic.item(),
        "omega": omega.item(),
        "zeta":  zeta.item(),
    }

# Dados de treinamento
t_data_t     = torch.tensor(t_train.reshape(-1, 1),     device=DEVICE, requires_grad=True)
x_star_t     = torch.tensor(x_train.reshape(-1, 1),     device=DEVICE)
xddot_star_t = torch.tensor(xddot_train.reshape(-1, 1), device=DEVICE)

# Pontos de colocação para o resíduo físico (período completo)
t_phys_np = np.linspace(0, T_TRAIN, 300, dtype=np.float32).reshape(-1, 1)
t_phys_t  = torch.tensor(t_phys_np, device=DEVICE, requires_grad=True)

def train_model(lam, label):
    model     = PINN_SDOF(hidden=50, n_layers=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-5
    )

    hist = {"total": [], "omega": [], "zeta": []}

    print(f"\n{'─'*65}")
    print(f"Treinando: {label}  (λ = {lam})")
    print(f"{'─'*65}")
    print(f"{'Época':>8}  {'Loss':>10}  {'L_data':>10}  {'L_phys':>10}  "
          f"{'ω pred':>8}  {'ζ pred':>8}")
    print(f"{'─'*65}")

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss, parts = compute_loss(
            model, t_data_t, x_star_t, xddot_star_t, t_phys_t, lam=lam
        )
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        hist["total"].append(loss.item())
        hist["omega"].append(parts["omega"])
        hist["zeta"].append(parts["zeta"])

        if epoch % 2000 == 0 or epoch == 1:
            print(f"{epoch:>8}  {loss.item():>10.4e}  {parts['data']:>10.4e}  "
                  f"{parts['phys']:>10.4e}  {parts['omega']:>8.4f}  {parts['zeta']:>8.5f}")

    omega_f = model.omega.item()
    zeta_f  = model.zeta.item()
    err_omega = abs(omega_f - OMEGA_TRUE) / OMEGA_TRUE * 100
    err_zeta  = abs(zeta_f  - ZETA_TRUE)  / ZETA_TRUE  * 100

    print(f"\n  {'Parâmetro':<12} {'Real':>10} {'Identificado':>14} {'Erro':>8}")
    print(f"  {'ω (rad/s)':<12} {OMEGA_TRUE:>10.4f} {omega_f:>14.4f} {err_omega:>7.2f}%")
    print(f"  {'ζ':<12} {ZETA_TRUE:>10.5f} {zeta_f:>14.5f} {err_zeta:>7.2f}%")

    return model, hist

model_pinn, hist_pinn = train_model(lam=1.0, label="PINN  (λ=1)")
model_ann,  hist_ann  = train_model(lam=0.0, label="ANN   (λ=0)")


t_full = torch.tensor(t_all.reshape(-1, 1), device=DEVICE)
with torch.no_grad():
    x_pinn = model_pinn(t_full).cpu().numpy().flatten()
    x_ann  = model_ann(t_full).cpu().numpy().flatten()

omega_pinn = model_pinn.omega.item()
zeta_pinn  = model_pinn.zeta.item()
omega_ann  = model_ann.omega.item()
zeta_ann   = model_ann.zeta.item()

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("PINN vs ANN — Sistema SDOF (identificação de ω e ζ)", fontsize=13, fontweight="bold")

# ── Gráfico 1: Predição de deslocamento ──────────────────
ax = axes[0, 0]
ax.axvspan(0, T_TRAIN, alpha=0.07, color="steelblue", label="Período de treino")
ax.plot(t_all, x_clean, "k-", lw=2.5, label="Analítica (real)")
ax.plot(t_all, x_pinn,  "r--", lw=1.8,
        label=f"PINN  (ω={omega_pinn:.3f}, ζ={zeta_pinn:.4f})")
ax.plot(t_all, x_ann,   "g:",  lw=1.8,
        label=f"ANN   (ω={omega_ann:.3f}, ζ={zeta_ann:.4f})")
ax.scatter(t_train[::5], x_train[::5], s=12, color="steelblue",
           alpha=0.5, zorder=3, label="Dados medidos (ruído 2%)")
ax.axvline(T_TRAIN, color="gray", lw=1, linestyle="--")
ax.set_xlabel("t [s]");  ax.set_ylabel("x(t) [m]")
ax.set_title("Deslocamento: treino e predição além de T_train")
ax.legend(fontsize=8);  ax.grid(True, alpha=0.35)

# ── Gráfico 2: Convergência de ω ─────────────────────────
ax = axes[0, 1]
ep = np.arange(1, EPOCHS + 1)
ax.axhline(OMEGA_TRUE, color="k", lw=2, label=f"ω real = {OMEGA_TRUE:.3f}")
ax.plot(ep, hist_pinn["omega"], "r-", lw=1.5, label="PINN")
ax.plot(ep, hist_ann["omega"],  "g-", lw=1.5, label="ANN",  alpha=0.7)
ax.set_xlabel("Época");  ax.set_ylabel("ω [rad/s]")
ax.set_title("Convergência de ω")
ax.legend(fontsize=9);   ax.grid(True, alpha=0.35)

# ── Gráfico 3: Convergência de ζ ─────────────────────────
ax = axes[1, 0]
ax.axhline(ZETA_TRUE, color="k", lw=2, label=f"ζ real = {ZETA_TRUE:.4f}")
ax.plot(ep, hist_pinn["zeta"], "r-", lw=1.5, label="PINN")
ax.plot(ep, hist_ann["zeta"],  "g-", lw=1.5, label="ANN",  alpha=0.7)
ax.set_xlabel("Época");  ax.set_ylabel("ζ")
ax.set_title("Convergência de ζ  (amortecimento)")
ax.legend(fontsize=9);   ax.grid(True, alpha=0.35)

# ── Gráfico 4: Histórico de perda ────────────────────────
ax = axes[1, 1]
ax.semilogy(hist_pinn["total"], "r-", lw=1.5, label="PINN (λ=1)")
ax.semilogy(hist_ann["total"],  "g-", lw=1.5, label="ANN  (λ=0)")
ax.set_xlabel("Época");  ax.set_ylabel("Loss")
ax.set_title("Histórico de treinamento")
ax.legend(fontsize=9);   ax.grid(True, which="both", alpha=0.35)

plt.tight_layout()

import os
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pinn_sdof.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGráfico salvo em outputs/pinn_sdof.png")