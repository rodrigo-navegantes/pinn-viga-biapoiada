import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

L   = 1.0        
EI  = 1.0        
q   = 1.0        

N_col  = 200      # 200 pontos distribuidos ao longo de x ∈ [0, 1]   
EPOCHS = 5_000    # número de iterações
LR     = 1e-3     # learning rate: controla o tamanho do passo que o otimizador dá a cada época para ajustar os pesos da rede. 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)

class PINN(nn.Module):
    """
    MLP simples:  1 → [64, 64, 64, 64] → 1
    Ativação: Tanh 
    """
    def __init__(self, hidden: int = 64, layers: int = 4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def nth_derivative(f: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    df = f
    for _ in range(n):
        df, = torch.autograd.grad(
            df, x,
            grad_outputs=torch.ones_like(df),
            create_graph=True,   # necessário para otimização
            retain_graph=True,
        )
    return df

def compute_loss(
    model: PINN,
    x_col: torch.Tensor,   # pontos de colocação (interior)
    x_bc:  torch.Tensor,   # pontos de contorno
) -> tuple[torch.Tensor, dict]:

    w_col   = model(x_col)
    d4w_dx4 = nth_derivative(w_col, x_col, 4)
    residual = EI * d4w_dx4 - q
    loss_pde = torch.mean(residual ** 2)

    x0 = torch.tensor([[0.0]], device=DEVICE, requires_grad=True)
    xL = torch.tensor([[L]],   device=DEVICE, requires_grad=True)

    w0  = model(x0)
    wL  = model(xL)
    d2w_x0 = nth_derivative(model(x0), x0, 2)
    d2w_xL = nth_derivative(model(xL), xL, 2)

    loss_bc = (
        w0  ** 2 +          # w(0)  = 0
        wL  ** 2 +          # w(L)  = 0
        d2w_x0 ** 2 +       # w''(0)= 0
        d2w_xL ** 2         # w''(L)= 0
    ).mean()

    loss_total = loss_pde + 10.0 * loss_bc

    return loss_total, {"pde": loss_pde.item(), "bc": loss_bc.item()}


def w_analytical(x_np: np.ndarray) -> np.ndarray:
    return (q / (24.0 * EI)) * x_np * (L**3 - 2*L*x_np**2 + x_np**3)


model     = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=500, factor=0.5, min_lr=1e-5
)

x_col_np = np.linspace(0, L, N_col).reshape(-1, 1).astype(np.float32)
x_col    = torch.tensor(x_col_np, device=DEVICE, requires_grad=True)
x_bc     = None   

history = {"total": [], "pde": [], "bc": []}

print(f"\n{'Epoch':>8}  {'Loss total':>12}  {'L_pde':>10}  {'L_bc':>10}  {'LR':>8}")
print("-" * 58)

for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    loss, parts = compute_loss(model, x_col, x_bc)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    history["total"].append(loss.item())
    history["pde"].append(parts["pde"])
    history["bc"].append(parts["bc"])

    if epoch % 1000 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>8}  {loss.item():>12.4e}  "
              f"{parts['pde']:>10.4e}  {parts['bc']:>10.4e}  {lr_now:>8.1e}")

print("\nTreinamento concluído!")

model.eval()

x_test_np = np.linspace(0, L, 300, dtype=np.float32).reshape(-1, 1)
x_test    = torch.tensor(x_test_np, device=DEVICE)

with torch.no_grad():
    w_pred_np = model(x_test).cpu().numpy().flatten()

w_exact_np = w_analytical(x_test_np.flatten())
error_np   = np.abs(w_pred_np - w_exact_np)
rel_error  = np.linalg.norm(error_np) / np.linalg.norm(w_exact_np)
print(f"\nErro relativo L2: {rel_error:.4e}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("PINN – Viga de Euler-Bernoulli Biapoiada", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(x_test_np, w_exact_np * 1e3, "k-",  lw=2.5, label="Analítica")
ax.plot(x_test_np, w_pred_np  * 1e3, "r--", lw=2,   label="PINN")
ax.set_xlabel("x [m]")
ax.set_ylabel("w [mm]")
ax.set_title("Deslocamento transversal")
ax.legend()
ax.grid(True, alpha=0.35)

ax = axes[1]
ax.semilogy(x_test_np, error_np, "b-", lw=1.8)
ax.set_xlabel("x [m]")
ax.set_ylabel("|w_PINN − w_exata|")
ax.set_title(f"Erro absoluto  (Rel. L2 = {rel_error:.2e})")
ax.grid(True, which="both", alpha=0.35)

ax = axes[2]
ax.semilogy(history["total"], "k-",  lw=1.5, label="Total")
ax.set_xlabel("Época")
ax.set_ylabel("MSE")
ax.set_title("Histórico de treinamento")
ax.legend()
ax.grid(True, which="both", alpha=0.35)

plt.tight_layout()
plt.savefig("pinn_euler_bernoulli.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico salvo em pinn_euler_bernoulli.png")
