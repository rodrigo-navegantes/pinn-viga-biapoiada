import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)

L       = 0.70                 # comprimento total [m]
b       = 0.02                 # largura [m]
h       = 0.01                 # espessura [m]
rho     = 2700.0               # densidade [kg/m³]
E_TRUE  = 70e9                 # módulo de Young real [Pa]
ETA_TRUE = 0.01                # fator de perda real η = Ei/Er
Ei_TRUE = ETA_TRUE * E_TRUE    # módulo imaginário real [Pa]
omega   = 2 * np.pi * 500.0    # frequência de análise [rad/s]
F_amp   = 1.0                  # amplitude da força [N]
x_force = 0.10                 # posição da força [m] (100 mm da base)

S = b * h                      # área da seção [m²]
I = b * h**3 / 12              # inércia à flexão [m⁴]

x_min = 0.24                   # [m]
x_max = 0.62                   # [m]

print("=" * 55)
print("PARÂMETROS DO PROBLEMA")
print("=" * 55)
print(f"  Viga: {L*1e3:.0f}×{b*1e3:.0f}×{h*1e3:.0f} mm  (alumínio)")
print(f"  I = {I:.4e} m⁴   S = {S:.4e} m²")
print(f"  E_real = {E_TRUE/1e9:.1f} GPa")
print(f"  η_true = {ETA_TRUE*100:.1f}%  →  Ei = {Ei_TRUE/1e9:.3f} GPa")
print(f"  ω = {omega/(2*np.pi):.1f} Hz")
print(f"  Zona de interesse: [{x_min:.2f}, {x_max:.2f}] m")


# 2.  GERAÇÃO DOS DADOS SINTÉTICOS

def cantilever_modes(n_modes, L, E, rho, S, I):
    beta_L = np.array([1.87510, 4.69409, 7.85476, 10.99554, 14.13717])
    if n_modes > 5:
        extra = np.array([(2*n - 1) * np.pi / 2 for n in range(6, n_modes + 1)])
        beta_L = np.concatenate([beta_L, extra])
    beta_L = beta_L[:n_modes]

    betas = beta_L / L
    omega_n = betas**2 * np.sqrt(E * I / (rho * S))
    return betas, omega_n


def mode_shape(x, beta, L):
    bL = beta * L
    sigma = (np.cos(bL) + np.cosh(bL)) / (np.sin(bL) + np.sinh(bL))
    phi = (np.cosh(beta * x) - np.cos(beta * x)
           - sigma * (np.sinh(beta * x) - np.sin(beta * x)))
    return phi


def compute_response(x_eval, x_force, omega_exc, E_complex, rho, S, I, L,
                     F_amp=1.0, n_modes=20):
    Er = np.real(E_complex)
    betas, omega_n_real = cantilever_modes(n_modes, L, Er, rho, S, I)

    omega_n2_complex = omega_n_real**2 * (E_complex / Er)

    w_complex = np.zeros(len(x_eval), dtype=complex)

    for n in range(n_modes):
        phi_x  = mode_shape(x_eval,   betas[n], L)
        phi_xf = mode_shape(np.array([x_force]), betas[n], L)[0]

        mu_n = rho * S * L

        denom = mu_n * (omega_n2_complex[n] - omega_exc**2)

        w_complex += phi_x * phi_xf * F_amp / denom

    return w_complex


N_total  = 500     # pontos para a curva completa
N_col    = 1024    # pontos de colocação (física) [Aqui mudei de 1024 para 256 pois acho o valor original do artigo alto demais]
N_obs    = 20      # pontos de observação (dados, conforme paper)

x_full = np.linspace(x_min, x_max, N_total)
E_complex_true = E_TRUE * (1 + 1j * ETA_TRUE)

print("\nCalculando resposta analítica...")
w_full = compute_response(x_full, x_force, omega, E_complex_true,
                          rho, S, I, L, F_amp)

wr_full = np.real(w_full)
wi_full = np.imag(w_full)

print(f"  |wr| max = {np.max(np.abs(wr_full))*1e6:.3f} μm")
print(f"  |wi| max = {np.max(np.abs(wi_full))*1e6:.3f} μm")
print(f"  Razão |wi|/|wr| = {np.max(np.abs(wi_full))/np.max(np.abs(wr_full)):.4f}")

W_SCALE = max(np.max(np.abs(wr_full)), np.max(np.abs(wi_full)))
wr_norm = wr_full / W_SCALE
wi_norm = wi_full / W_SCALE

X_MIN, X_MAX = x_min, x_max
X_SCALE = (X_MAX - X_MIN) / 2.0
X_CENTER = (X_MAX + X_MIN) / 2.0

def normalize_x(x):
    return (x - X_CENTER) / X_SCALE

def denormalize_x(xi):
    return xi * X_SCALE + X_CENTER

xi_full = normalize_x(x_full)  # ∈ [-1, 1]

obs_idx = np.linspace(0, N_total - 1, N_obs, dtype=int)
x_obs   = x_full[obs_idx]
xi_obs  = xi_full[obs_idx]
wr_obs  = wr_norm[obs_idx]
wi_obs  = wi_norm[obs_idx]

xi_col  = np.linspace(-1.0, 1.0, N_col, dtype=np.float32)

print(f"\nDados preparados:")
print(f"  W_SCALE = {W_SCALE:.4e} m")
print(f"  Obs. pts: {N_obs}  |  Coloc. pts: {N_col}")


# 3.  FATOR DE NORMALIZAÇÃO DA EDP
kappa = I / (rho * S * omega**2 * X_SCALE**4)
print(f"\nFator κ = I/(ρSω²xs⁴) = {kappa:.4e}")
print(f"  Er·κ = {E_TRUE * kappa:.4f}  (deve ser O(1) para EDP balanceada)")
print(f"  Ei·κ = {Ei_TRUE * kappa:.6f}")

E_NET_SCALE = 1e11   # Pa


# 4.  ARQUITETURA DAS REDES (9 camadas, 10 neurônios)
class BeamNet(nn.Module):
    def __init__(self, n_layers=9, n_neurons=10):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xi):
        return self.net(xi)


class ENet(nn.Module):
    def __init__(self, init_value_Pa):
        super().__init__()

        self.bias = nn.Parameter(
            torch.tensor(init_value_Pa / E_NET_SCALE, dtype=torch.float32)
        )

    def forward(self):
        return self.bias * E_NET_SCALE


GAMMA = 1.0    # peso dos dados
BETA  = 1e-2   # peso da física (conforme paper)
KAPPA = float(kappa)

def nth_deriv(f, x, n=1):
    """
    Calcula a derivada de ordem n de f em relação a x usando autograd.
    """
    df = f

    for _ in range(n):
        df, = torch.autograd.grad(
            outputs=df,
            inputs=x,
            grad_outputs=torch.ones_like(df),
            create_graph=True,
            retain_graph=True,
        )

    return df

def compute_loss(net_wr, net_wi, E_r_net, E_i_net,
                 xi_obs_t, wr_obs_t, wi_obs_t,
                 xi_col_t):
    Er = E_r_net.bias * E_NET_SCALE   # [Pa]
    Ei = E_i_net.bias * E_NET_SCALE   # [Pa]

    wr_pred_obs = net_wr(xi_obs_t)
    wi_pred_obs = net_wi(xi_obs_t)
    Rd = (torch.mean((wr_pred_obs - wr_obs_t)**2) +
          torch.mean((wi_pred_obs - wi_obs_t)**2))

    wr_col = net_wr(xi_col_t)
    wi_col = net_wi(xi_col_t)

    d4wr = nth_deriv(wr_col, xi_col_t, n=4)
    d4wi = nth_deriv(wi_col, xi_col_t, n=4)

    if ep == 1:
        print("\n===== Escalas da EDP =====")

        print(f"|wr|        = {wr_col.abs().mean().item():.3e}")
        print(f"|wi|        = {wi_col.abs().mean().item():.3e}")

        print(f"|d4wr|      = {d4wr.abs().mean().item():.3e}")
        print(f"|d4wi|      = {d4wi.abs().mean().item():.3e}")

        print(f"|κErd4|     = {(KAPPA*Er*d4wr).abs().mean().item():.3e}")
        print(f"|κEid4|     = {(KAPPA*Ei*d4wi).abs().mean().item():.3e}")

    Rf_real = torch.mean(
        (KAPPA * (Er * d4wr - Ei * d4wi) / W_SCALE - wr_col)**2
    )

    Rf_imag = torch.mean(
        (KAPPA * (Er * d4wi + Ei * d4wr) / W_SCALE - wi_col)**2
    )

    loss_total = GAMMA * Rd + BETA * (Rf_real + Rf_imag)

    return loss_total, {
        "Rd":      Rd.item(),
        "Rf_real": Rf_real.item(),
        "Rf_imag": Rf_imag.item(),
        "Er":      Er.item(),
        "Ei":      Ei.item(),
        "eta":     (Ei / Er).item(),
    }


# 6.  INICIALIZAÇÃO
ER_INIT  = 110e9
EI_INIT  = 0.10 * ER_INIT   # η = 10%

net_wr  = BeamNet(n_layers=9, n_neurons=10)
net_wi  = BeamNet(n_layers=9, n_neurons=10)
E_r_net = ENet(ER_INIT)
E_i_net = ENet(EI_INIT)

print(f"\nChutes iniciais:")
print(f"  Er_init = {ER_INIT/1e9:.1f} GPa  (real = {E_TRUE/1e9:.1f} GPa)")
print(f"  η_init  = {EI_INIT/ER_INIT*100:.1f}%   (real = {ETA_TRUE*100:.1f}%)")

xi_obs_t = torch.tensor(xi_obs.reshape(-1, 1),   dtype=torch.float32)
wr_obs_t = torch.tensor(wr_obs.reshape(-1, 1),   dtype=torch.float32)
wi_obs_t = torch.tensor(wi_obs.reshape(-1, 1),   dtype=torch.float32)
xi_col_t = torch.tensor(xi_col.reshape(-1, 1),   dtype=torch.float32,
                        requires_grad=True)

# 7.  TREINAMENTO
EPOCHS = 30_000

all_params = (list(net_wr.parameters()) +
              list(net_wi.parameters()) +
              list(E_r_net.parameters()) +
              list(E_i_net.parameters()))

optimizer = torch.optim.Adam(all_params, lr=5e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-3,
    total_steps=EPOCHS,
    pct_start=0.10,       # sobe nos primeiros 10%
    anneal_strategy="cos",
    div_factor=10.0,      # lr inicial = max_lr / div_factor
    final_div_factor=100.0,
)

hist = {k: [] for k in ["tot", "Rd", "Rf_real", "Rf_imag", "Er", "Ei", "eta"]}

print(f"\n{'─'*72}")
print(f"{'Época':>7}  {'Loss':>10}  {'Rd':>10}  {'Rf_r':>10}  "
      f"{'Er [GPa]':>10}  {'η [%]':>8}")
print(f"{'─'*72}")

for ep in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    loss, p = compute_loss(net_wr, net_wi, E_r_net, E_i_net,
                           xi_obs_t, wr_obs_t, wi_obs_t, xi_col_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()

    for k in hist:
        hist[k].append(p[k] if k != "tot" else loss.item())

    if ep % 3000 == 0 or ep == 1:
        print(f"{ep:>7}  {loss.item():>10.3e}  {p['Rd']:>10.3e}  "
              f"{p['Rf_real']:>10.3e}  {p['Er']/1e9:>10.4f}  "
              f"{p['eta']*100:>8.4f}")

Er_f  = E_r_net.value().item()
Ei_f  = E_i_net.value().item()
eta_f = Ei_f / Er_f

print(f"\n{'═'*55}")
print(f"  {'':12} {'Real':>10} {'PINN':>12} {'NMSE [%]':>10}")
print(f"  {'─'*53}")
print(f"  {'Er [GPa]':<12} {E_TRUE/1e9:>10.4f} {Er_f/1e9:>12.4f} "
      f"{abs(Er_f-E_TRUE)/E_TRUE*100:>9.3f}%")
print(f"  {'η [%]':<12} {ETA_TRUE*100:>10.4f} {eta_f*100:>12.4f} "
      f"{abs(eta_f-ETA_TRUE)/ETA_TRUE*100:>9.3f}%")
print(f"  {'Ei [GPa]':<12} {Ei_TRUE/1e9:>10.4f} {Ei_f/1e9:>12.6f} "
      f"{abs(Ei_f-Ei_TRUE)/Ei_TRUE*100:>9.3f}%")
print(f"{'═'*55}")

xi_full_t = torch.tensor(xi_full.reshape(-1, 1), dtype=torch.float32)

with torch.no_grad():
    wr_pred = net_wr(xi_full_t).numpy().flatten() * W_SCALE
    wi_pred = net_wi(xi_full_t).numpy().flatten() * W_SCALE

gen_mask = np.ones(N_total, dtype=bool)
gen_mask[obs_idx] = False
x_gen  = x_full[gen_mask]
wr_gen = wr_full[gen_mask]
wi_gen = wi_full[gen_mask]
wr_pred_gen = wr_pred[gen_mask]
wi_pred_gen = wi_pred[gen_mask]

nmse_wr = np.mean((wr_pred_gen - wr_gen)**2) / np.mean(wr_gen**2) * 100
nmse_wi = np.mean((wi_pred_gen - wi_gen)**2) / np.mean(wi_gen**2) * 100
print(f"\nNMSE (generalização):")
print(f"  wr: {nmse_wr:.4f}%")
print(f"  wi: {nmse_wi:.4f}%")

ep_arr = np.arange(1, EPOCHS + 1)

fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    f"PINN – Euler-Bernoulli Dinâmica: Identificação de Er e η\n"
    f"Er = {Er_f/1e9:.3f} GPa  |  η = {eta_f*100:.4f}%  "
    f"(reais: {E_TRUE/1e9:.1f} GPa, {ETA_TRUE*100:.1f}%)",
    fontsize=13, fontweight="bold"
)
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(x_full*1e3, wr_full*1e6, "r-", lw=2,
         label="Analítico (generalização)")
ax0.plot(x_full*1e3, wr_pred*1e6, "b-", lw=1.5,
         label=f"PINN (NMSE={nmse_wr:.2f}%)")
ax0.scatter(x_obs*1e3, wr_obs*W_SCALE*1e6, s=40,
            color="red", zorder=5, label=f"Obs. pts ({N_obs})")
ax0.set_xlabel("x [mm]"); ax0.set_ylabel("wr [μm]")
ax0.set_title("Deslocamento real ŵr(x)")
ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(x_full*1e3, wi_full*1e6, "r-", lw=2,
         label="Analítico (generalização)")
ax1.plot(x_full*1e3, wi_pred*1e6, "b-", lw=1.5,
         label=f"PINN (NMSE={nmse_wi:.2f}%)")
ax1.scatter(x_obs*1e3, wi_obs*W_SCALE*1e6, s=40,
            color="red", zorder=5, label=f"Obs. pts ({N_obs})")
ax1.set_xlabel("x [mm]"); ax1.set_ylabel("wi [μm]")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax1.set_title("Deslocamento imaginário ŵi(x)  [proporcional ao amortecimento]")

ax2 = fig.add_subplot(gs[1, 0])
ax2.axhline(E_TRUE/1e9,  color="k", lw=2, label=f"Er real = {E_TRUE/1e9:.1f} GPa")
ax2.axhline(ER_INIT/1e9, color="gray", lw=1, ls=":",
            label=f"Er inicial = {ER_INIT/1e9:.0f} GPa")
ax2.plot(ep_arr, np.array(hist["Er"])/1e9, "b-", lw=1.5,
         label="Er identificado")
ax2.set_xlabel("Época"); ax2.set_ylabel("Er [GPa]")
ax2.set_title("Convergência: Módulo de Young Er")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.axhline(ETA_TRUE*100, color="k", lw=2,
            label=f"η real = {ETA_TRUE*100:.1f}%")
ax3.axhline(EI_INIT/ER_INIT*100, color="gray", lw=1, ls=":",
            label=f"η inicial = {EI_INIT/ER_INIT*100:.0f}%")
ax3.plot(ep_arr, np.array(hist["eta"])*100, "r-", lw=1.5,
         label="η identificado")
ax3.set_xlabel("Época"); ax3.set_ylabel("η [%]")
ax3.set_title("Convergência: Fator de perda η = Ei/Er")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, :])
ax4.semilogy(ep_arr, hist["tot"],    "k-",  lw=1.8, label="Loss total")
ax4.semilogy(ep_arr, hist["Rd"],     "b--", lw=1.4, label="Rd (dados)")
ax4.semilogy(ep_arr, hist["Rf_real"],"r-.", lw=1.2, label="Rf_real (EDP real)")
ax4.semilogy(ep_arr, hist["Rf_imag"],"g-.", lw=1.2, label="Rf_imag (EDP imag)")
ax4.set_xlabel("Época"); ax4.set_ylabel("Loss (log)")
ax4.set_title("Histórico de treinamento")
ax4.legend(fontsize=9, ncol=4); ax4.grid(True, which="both", alpha=0.3)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pinn_euler_bernoulli_dinamica.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("\nGráfico salvo em outputs/pinn_euler_bernoulli_dinamica.png")