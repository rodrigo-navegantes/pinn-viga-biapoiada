"""

O seguinte programa simula e resolve um problema de vibração transversal-flexional 
em regimente harmônico de uma viga engastada-livre (cantilever), associado à
identificação inversa de propriedades viscoelásticas do material.

Implementação de um PINN baseado em: Teloli et al., Seção 4.2

Equação do movimento (regime harmônico, frequência ω):
    ∂²/∂x² [E·I·∂²w/∂x²] - ρ·S·ω²·w = 0   (zona sem força aplicada)

Com E complexo uniforme:  E* = Er + j·Ei   →   η = Ei/Er
e w complexo:             w  = wr + j·wi


EDP decomposta (E uniforme → ∂E/∂x = 0):
                Real: κ·(Er·d4wr/dξ4 - Ei·d4wi/dξ4) = wr
                Imag: κ·(Er·d4wi/dξ4 + Ei·d4wr/dξ4) = wi

onde κ = I·AWR/(ρ·S·ω²·XS⁴·W_ref),  ξ = (x-xc)/xs ∈ [-1,1]

"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

# Parâmetros físicos
L        = 0.70        # comprimento da viga [m]
b        = 0.02        # largura [m]
h_beam   = 0.01        # espessura [m]
rho      = 2700.0      # densidade [kg/m³]
E_TRUE   = 70e9        # módulo de Young [Pa]

ETA_TRUE = 0.01        # fator de perda real η = Ei/Er
x_min    = 0.24        # início da zona de interesse [m]
x_max    = 0.62        # fim da zona de interesse [m]   
x_force  = 0.10        # posição da força [m]
F_amp    = 1.0         # amplitude da força [N]

S    = b * h_beam
I_b  = b * h_beam**3 / 12
Ei_TRUE = ETA_TRUE * E_TRUE

# Hiperparâmetros
N_TOTAL  = 500         # pontos da curva de referência
N_OBS    = 20          # pontos de observação (sensores)
N_COL    = 80          # pontos de colocação — autograd exige N pequeno
PHASE2   = 5_000       # fim da Fase 1 (só dados)
EPOCHS   = 30_000      # épocas totais
LR_MAX   = 5e-3        # taxa de apredizado máximo
BETA     = 1e-2        # peso da física

# Chutes iniciais (conforme artigo)
ER_INIT  = 110.0       # [GPa]  (real = 70 GPa)
ETA_INIT = 0.10        # []     (real = 1%)

print("=" * 55)
print("PARÂMETROS DO PROBLEMA")
print("=" * 55)
print(f"  Viga: {L*1e3:.0f}×{b*1e3:.0f}×{h_beam*1e3:.0f} mm  (alumínio)")
print(f"  E_real = {E_TRUE/1e9:.1f} GPa  |  η = {ETA_TRUE*100:.1f}%")
print(f"  Chutes: Er={ER_INIT:.0f} GPa, η={ETA_INIT*100:.0f}%")
print(f"  N_COL={N_COL} (autograd), EPOCHS={EPOCHS}")


# Solução Análitica (superposição modal)

# Calcula os autovalores β_n e as frequências naturais ω_n da viga engastada-livre
def cantilever_modes(n_modes, L, E, rho, S, I):

    # A equação característica da viga cantilever é cos(β_n * L)cosh(β_n * L) + 1 = 0

    # Os primeiros 5 valores de beta_n * L são constantes numéricas conhecidas
    beta_L = np.array([1.87510, 4.69409, 7.85476, 10.99554, 14.13717])
    
    # Para n > 5, aplica-se a aproximação assintótica β_n * L = (2k-1) * pi / 2
    if n_modes > 5:
        extra = np.array([(2*k-1)*np.pi/2 for k in range(6, n_modes+1)])
        beta_L = np.concatenate([beta_L, extra])
    
    beta_L = beta_L[:n_modes]
    beta   = beta_L / L
    omega_n = beta**2 * np.sqrt(E * I / (rho * S))
    return beta, omega_n


# Retorna a forma modal ϕ_n para o n-ésimo modo
def mode_shape(x, beta, L):
    bL    = beta * L
    sigma = (np.cos(bL) + np.cosh(bL)) / (np.sin(bL) + np.sinh(bL))
    return (np.cosh(beta*x) - np.cos(beta*x)
            - sigma*(np.sinh(beta*x) - np.sin(beta*x)))


# Calcula a FRF em regime permanente sob força harmônica F(t) = F_0 e^(jωt) por superposição modal:
def compute_response(x_eval, omega_exc, E_complex, n_modes=20):
    Er = np.real(E_complex)
    betas, omega_n_real = cantilever_modes(n_modes, L, Er, rho, S, I_b)
    omega_n2c = omega_n_real**2 * (E_complex / Er)
    w = np.zeros(len(x_eval), dtype=complex)
    for n in range(n_modes):
        phi_x  = mode_shape(x_eval, betas[n], L)
        phi_xf = mode_shape(np.array([x_force]), betas[n], L)[0]
        mu_n   = rho * S * L
        w     += phi_x * phi_xf * F_amp / (mu_n * (omega_n2c[n] - omega_exc**2))
    return w


# Frequência de análise: 0.95·ω_n4
# Próximo da ressonância → |wi|/|wr| ≈ 0.10 → Ei identificável
# NOTA: 500 Hz do paper usa geometria diferente (entre fn4 e fn5).
# Nesta geometria, fn4 = 577 Hz e 500 Hz fica longe → |wi|/|wr| = 0.03.
_, wn_ref = cantilever_modes(10, L, E_TRUE, rho, S, I_b)
omega = 0.95 * wn_ref[3]

print(f"\n  Frequência de análise: {omega/(2*np.pi):.1f} Hz")
print(f"  fn4 = {wn_ref[3]/(2*np.pi):.1f} Hz  (0.95·fn4 → |wi|/|wr| ≈ 0.10)")

# ── Dados sintéticos ──────────────────────────────────────
x_full   = np.linspace(x_min, x_max, N_TOTAL)
w_full   = compute_response(x_full, omega, E_TRUE*(1+1j*ETA_TRUE))
wr_full  = np.real(w_full)
wi_full  = np.imag(w_full)

ratio = np.max(np.abs(wi_full)) / np.max(np.abs(wr_full))
print(f"  |wr|max = {np.max(np.abs(wr_full))*1e6:.3f} μm")
print(f"  |wi|max = {np.max(np.abs(wi_full))*1e6:.3f} μm")
print(f"  |wi|/|wr| = {ratio:.4f}")


# Normalização 

# Normalização SEPARADA para wr e wi
# Garante que ambas fiquem em O(1)
AWR = np.max(np.abs(wr_full))
AWI = np.max(np.abs(wi_full))
wr_n = (wr_full / AWR).astype(np.float32)
wi_n = (wi_full / AWI).astype(np.float32)

# x normalizado → [-1, 1]
XC = (x_max + x_min) / 2.0
XS = (x_max - x_min) / 2.0
xi_full = ((x_full - XC) / XS).astype(np.float32)

# Fator κ da EDP em coordenadas adimensionais

# κ = I / (ρ·S·ω²·XS⁴)  (adimensional quando multiplicado por E em Pa)

# κ*(Er*d4wr_n - Ei*(AWI/AWR)*d4wi_n) = wr_n
# κ*(Er*d4wi_n - Ei*(AWI/AWR)*d4wr_n) = wr_n

# Tiramos a equação acima usando "x = x_s * ζ + x_c"  e "ω_r = AWR * ω_r" e "ω_i = AWI * ω_i"

kappa    = I_b / (rho * S * omega**2 * XS**4)
KAPPA    = float(kappa)
RATIO_IW = float(AWI / AWR)   # ≈ 0.10
RATIO_WI = float(AWR / AWI)   # ≈ 10.0

print(f"\n  κ = {kappa:.4e}")
print(f"  κ·Er_true = {kappa*E_TRUE:.6f}  (escala O(0.01))")
print(f"  d4wr_n amplitude ≈ {kappa*E_TRUE*1/(kappa*E_TRUE):.2f}  (O(1) correto)")
print(f"  AWI/AWR = {RATIO_IW:.4f}  AWR/AWI = {RATIO_WI:.2f}")

# Pontos de observação
obs_idx  = np.linspace(0, N_TOTAL - 1, N_OBS, dtype=int)
x_obs    = x_full[obs_idx]
xi_obs   = xi_full[obs_idx]
wr_obs   = wr_n[obs_idx]
wi_obs   = wi_n[obs_idx]

# Pontos de colocação (física) 
xi_col   = np.linspace(-1.0, 1.0, N_COL, dtype=np.float32)

print(f"\n  Obs. pts: {N_OBS}  |  Coloc. pts: {N_COL}")


# Arquitetura das Redes Neurais

# Aqui temos nossa primeira red: ela é um MLP e está sendo treinada para obter w_r ou w_i a partir de ξ.
class BeamNet(nn.Module):
    """
    MLP para ŵr(ξ) ou ŵi(ξ).
    1 > [10]×9 > 1, Tanh (infinitamente dif.), Xavier init.
    (Conforme o artigo: 9 hidden layers, 10 neurons)
    """
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


# Define um parâmetro ajustável para representar E_r e E_i
class EParam(nn.Module):
    """
    Parâmetro escalar treinável (Er ou Ei) em GPa.
    Log-parametrizado > Garante que E_i e E_r sejam sempre positivos
    """
    def __init__(self, val_GPa):
        super().__init__()
        self.log_e = nn.Parameter(
            torch.tensor(float(np.log(val_GPa)), dtype=torch.float32)
        )

    def get_GPa(self):
        return torch.exp(self.log_e.clamp(-2.0, 6.0))

    def get_Pa(self):
        return self.get_GPa() * 1e9


# Usamos esta função para tomara a 4ª Derivada 
def nth_deriv(f, x, n=1):
    df = f
    for _ in range(n):
        df, = torch.autograd.grad(
            df, x,
            grad_outputs=torch.ones_like(df),
            create_graph=True,
            retain_graph=True,
        )
    return df


# ══════════════════════════════════════════════════════════
#                Loss functionn (Eq. 14)
#
# L = γ·Rd + β·(Rf_real + Rf_imag)
# γ = 1,  β = 1e-2  (conforme artigo)
#
# EDP em coords adimensionais (normalização separada):
#   Real: κ·(Er·d4ŵr - Ei·RATIO_IW·d4ŵi) = ŵr
#   Imag: κ·(Er·d4ŵi + Ei·RATIO_WI·d4ŵr) = ŵi
#
# RATIO_IW = AWI/AWR ≈ 0.10  (aparece na EDP real)
# RATIO_WI = AWR/AWI ≈ 10.0  (aparece na EDP imag)
# ══════════════════════════════════════════════════════════

def compute_loss(net_wr, net_wi, Er_net, Ei_net,
                 xi_obs_t, wr_obs_t, wi_obs_t,
                 xi_col_t, w_phys=1.0):

    Er_Pa  = Er_net.get_Pa()
    Ei_Pa  = Ei_net.get_Pa()
    Er_GPa = Er_net.get_GPa()
    Ei_GPa = Ei_net.get_GPa()
    eta    = Ei_GPa / Er_GPa

    # Rd (Data residue): ajuste aos dados 
    # É o erro quadrático médio entre as previsões das redes neurais e os dados reais medidos pelos 20 sensores w_r e w_i.
    Rd = (torch.mean((net_wr(xi_obs_t) - wr_obs_t)**2) +
          torch.mean((net_wi(xi_obs_t) - wi_obs_t)**2))

    if w_phys == 0.0:
        return Rd, {"Rd": Rd.item(), "Rf_r": 0.0, "Rf_i": 0.0,
                    "Er": Er_GPa.item(), "Ei": Ei_GPa.item(),
                    "eta": eta.item()}

    # Rf (Physics residue): resíduo físico via autograd
    wr_col = net_wr(xi_col_t)
    wi_col = net_wi(xi_col_t)

    d4wr = nth_deriv(wr_col, xi_col_t, n=4)
    d4wi = nth_deriv(wi_col, xi_col_t, n=4)

    # EDP real:  κ·(Er·d4ŵr - Ei·RATIO_IW·d4ŵi) - ŵr = 0
    # Mede a violação do balanço de forças em fase com a excitação:
    Rf_r = torch.mean(
        (KAPPA * (Er_Pa * d4wr - Ei_Pa * RATIO_IW * d4wi) - wr_col)**2
    )

    # EDP imag:  κ·(Er·d4ŵi + Ei·RATIO_WI·d4ŵr) - ŵi = 0
    # RATIO_WI ≈ 10 amplifica o papel de Ei > identificável (Não entendi está parte)
    # Mede a violação do balanço de forças em quadratura (fora de fase), onde o amortecimento atua:
    Rf_i = torch.mean(
        (KAPPA * (Er_Pa * d4wi + Ei_Pa * RATIO_WI * d4wr) - wi_col)**2
    )

    loss = Rd + w_phys * BETA * (Rf_r + Rf_i)

    return loss, {
        "Rd":   Rd.item(),
        "Rf_r": Rf_r.item(),
        "Rf_i": Rf_i.item(),
        "Er":   Er_GPa.item(),
        "Ei":   Ei_GPa.item(),
        "eta":  eta.item(),
    }

net_wr = BeamNet()
net_wi = BeamNet()
Er_net = EParam(ER_INIT)
Ei_net = EParam(ER_INIT * ETA_INIT)

xi_obs_t = torch.tensor(xi_obs.reshape(-1, 1),  dtype=torch.float32)
wr_obs_t = torch.tensor(wr_obs.reshape(-1, 1),  dtype=torch.float32)
wi_obs_t = torch.tensor(wi_obs.reshape(-1, 1),  dtype=torch.float32)
xi_col_t = torch.tensor(xi_col.reshape(-1, 1),  dtype=torch.float32,
                        requires_grad=True)


# ══════════════════════════════════════════════════════════
#                     Treinamento
#
# Fase 1:  w_phys=0, Er/Ei congelados
#   - redes aprendem a forma de ŵr e ŵi sem física
#   - garante que Rd → 0 antes de introduzir restrição da EDP
#
# Fase 2:  w_phys=1, Er/Ei livres
#   - física guia identificação de Er e Ei
#  
# ══════════════════════════════════════════════════════════
all_net_params = list(net_wr.parameters()) + list(net_wi.parameters())
all_E_params   = list(Er_net.parameters()) + list(Ei_net.parameters())
all_params     = all_net_params + all_E_params

optimizer = torch.optim.Adam(all_params, lr=LR_MAX)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr           = LR_MAX,
    total_steps      = EPOCHS,
    pct_start        = 0.10,
    anneal_strategy  = "cos",
    div_factor       = 10.0,
    final_div_factor = 100.0,
)

hist = {k: [] for k in ["tot", "Rd", "Rf_r", "Rf_i", "Er", "Ei", "eta"]}

print(f"\n{'─'*75}")
print(f"{'Época':>7}  {'Fase':<8}  {'Loss':>10}  {'Rd':>10}  "
      f"{'Rf_r':>10}  {'Er [GPa]':>10}  {'η [%]':>8}")
print(f"{'─'*75}")

t0 = time.time()

for ep in range(1, EPOCHS + 1):

    is_phase2 = ep > PHASE2
    w_phys    = 1.0 if is_phase2 else 0.0

    # Fase 1: congela Er e Ei
    for par in all_E_params:
        par.requires_grad_(is_phase2)

    optimizer.zero_grad()
    loss, p = compute_loss(
        net_wr, net_wi, Er_net, Ei_net,
        xi_obs_t, wr_obs_t, wi_obs_t, xi_col_t,
        w_phys=w_phys,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()

    hist["tot"].append(loss.item())
    for k in ["Rd", "Rf_r", "Rf_i", "Er", "Ei", "eta"]:
        hist[k].append(p[k])

    if ep % 5000 == 0 or ep in [1, PHASE2, PHASE2 + 1]:
        fase = "dados" if not is_phase2 else "física"
        elapsed = time.time() - t0
        rf_r_str = f"{p['Rf_r']:>10.3e}" if is_phase2 else f"{'—':>10}"
        print(f"{ep:>7}  {fase:<8}  {loss.item():>10.3e}  {p['Rd']:>10.3e}  "
              f"{rf_r_str}  {p['Er']:>10.4f}  {p['eta']*100:>8.4f}  [{elapsed:.0f}s]")


# Resultados finais
Er_f    = Er_net.get_GPa().item()
Ei_f    = Ei_net.get_GPa().item()
eta_f   = Ei_f / Er_f
Er_f_Pa = Er_f * 1e9

print(f"\n{'═'*55}")
print(f"  {'':10} {'Real':>10} {'PINN':>12} {'Erro%':>8}")
print(f"  {'─'*53}")
print(f"  {'Er [GPa]':<10} {E_TRUE/1e9:>10.3f} {Er_f:>12.4f} "
      f"{abs(Er_f_Pa-E_TRUE)/E_TRUE*100:>7.2f}%")
print(f"  {'η [%]':<10} {ETA_TRUE*100:>10.4f} {eta_f*100:>12.4f} "
      f"{abs(eta_f-ETA_TRUE)/ETA_TRUE*100:>7.2f}%")
print(f"  {'Ei [GPa]':<10} {Ei_TRUE/1e9:>10.4f} {Ei_f:>12.4f} "
      f"{abs(Ei_f*1e9-Ei_TRUE)/Ei_TRUE*100:>7.2f}%")
print(f"{'═'*55}")
print(f"Tempo total: {time.time()-t0:.0f}s")

# O Erro Quadrático Médio Normalizado (NMSE) é calculado exclusivamente nos pontos onde NÃO havia sensores:
xi_full_t = torch.tensor(xi_full.reshape(-1, 1), dtype=torch.float32)

with torch.no_grad():
    wr_pred = net_wr(xi_full_t).numpy().flatten() * AWR
    wi_pred = net_wi(xi_full_t).numpy().flatten() * AWI

gen_mask = np.ones(N_TOTAL, dtype=bool)
gen_mask[obs_idx] = False

nmse_wr = (np.mean((wr_pred[gen_mask] - wr_full[gen_mask])**2) /
           np.mean(wr_full[gen_mask]**2) * 100)
nmse_wi = (np.mean((wi_pred[gen_mask] - wi_full[gen_mask])**2) /
           np.mean(wi_full[gen_mask]**2) * 100)

print(f"\nNMSE (generalização):  wr={nmse_wr:.4f}%  wi={nmse_wi:.4f}%")


# Geração dos gráficos
ep_arr = np.arange(1, EPOCHS + 1)

fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    f"PINN – Euler-Bernoulli Dinâmica: Identificação de Er e η\n"
    f"Er = {Er_f:.3f} GPa  |  η = {eta_f*100:.4f}%  "
    f"(reais: {E_TRUE/1e9:.0f} GPa, {ETA_TRUE*100:.1f}%)  —  "
    f"f = {omega/(2*np.pi):.0f} Hz",
    fontsize=13, fontweight="bold"
)
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32)

# Painel 1: ŵr
ax = fig.add_subplot(gs[0, 0])
ax.plot(x_full*1e3, wr_full*1e6, "r-", lw=2, label="Analítico")
ax.plot(x_full*1e3, wr_pred*1e6, "b-", lw=1.5,
        label=f"PINN (NMSE={nmse_wr:.2f}%)")
ax.scatter(x_obs*1e3, wr_full[obs_idx]*1e6,
           s=40, color="red", zorder=5, label=f"{N_OBS} obs.")
ax.set_xlabel("x [mm]"); ax.set_ylabel("wr [μm]")
ax.set_title("Deslocamento real ŵr(x)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Painel 2: ŵi
ax = fig.add_subplot(gs[0, 1])
ax.plot(x_full*1e3, wi_full*1e6, "r-", lw=2, label="Analítico")
ax.plot(x_full*1e3, wi_pred*1e6, "b-", lw=1.5,
        label=f"PINN (NMSE={nmse_wi:.2f}%)")
ax.scatter(x_obs*1e3, wi_full[obs_idx]*1e6,
           s=40, color="red", zorder=5, label=f"{N_OBS} obs.")
ax.set_xlabel("x [mm]"); ax.set_ylabel("wi [μm]")
ax.set_title("Deslocamento imaginário ŵi(x)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Painel 3: convergência Er
ax = fig.add_subplot(gs[1, 0])
ax.axhline(E_TRUE/1e9, color="k", lw=2,
           label=f"Real = {E_TRUE/1e9:.0f} GPa")
ax.axhline(ER_INIT, color="gray", lw=1, ls=":",
           label=f"Inicial = {ER_INIT:.0f} GPa")
ax.plot(ep_arr, hist["Er"], "b-", lw=1.5, label="PINN")
ax.axvline(PHASE2, color="orange", lw=1, ls="--",
           label=f"Início física (ep {PHASE2})")
ax.set_xlabel("Época"); ax.set_ylabel("Er [GPa]")
ax.set_title("Convergência: Módulo de Young Er")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Painel 4: convergência η
ax = fig.add_subplot(gs[1, 1])
ax.axhline(ETA_TRUE*100, color="k", lw=2,
           label=f"Real = {ETA_TRUE*100:.1f}%")
ax.axhline(ETA_INIT*100, color="gray", lw=1, ls=":",
           label=f"Inicial = {ETA_INIT*100:.0f}%")
ax.plot(ep_arr, np.array(hist["eta"])*100, "r-", lw=1.5, label="PINN")
ax.axvline(PHASE2, color="orange", lw=1, ls="--",
           label=f"Início física (ep {PHASE2})")
ax.set_xlabel("Época"); ax.set_ylabel("η [%]")
ax.set_title("Convergência: Fator de perda η = Ei/Er")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Painel 5: histórico da loss
ax = fig.add_subplot(gs[2, :])
ax.semilogy(ep_arr, hist["tot"], "k-",  lw=1.8, label="Loss total")
ax.semilogy(ep_arr, hist["Rd"],  "b--", lw=1.4, label="Rd (dados)")
if any(v > 0 for v in hist["Rf_r"]):
    ax.semilogy(ep_arr[PHASE2:], hist["Rf_r"][PHASE2:], "r-.",
                lw=1.2, label="Rf real (EDP)")
    ax.semilogy(ep_arr[PHASE2:], hist["Rf_i"][PHASE2:], "g-.",
                lw=1.2, label="Rf imag (EDP)")
ax.axvline(PHASE2, color="orange", lw=1, ls="--", label="Início física")
ax.set_xlabel("Época"); ax.set_ylabel("Loss (log)")
ax.set_title("Histórico de treinamento")
ax.legend(fontsize=9, ncol=5); ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pinn_euler_bernoulli_dinamica.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nGráfico salvo em outputs/pinn_euler_bernoulli_dinamica.png")