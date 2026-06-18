# PINN – Viga de Euler-Bernoulli Biapoiada

Implementação de **Physics-Informed Neural Networks (PINNs)** em PyTorch para resolver a equação diferencial da viga de Euler-Bernoulli, abordando tanto o **problema direto** quanto o **problema inverso** de identificação de parâmetros estruturais.

## Equação governante

A equação de Euler-Bernoulli para uma viga sob carga distribuída estática é:

```text
EI · d⁴w/dx⁴ = q
```

onde:

| Símbolo | Descrição                                               | Unidade |
| ------- | ------------------------------------------------------- | ------- |
| `w(x)`  | Deslocamento transversal                                | m       |
| `EI`    | Rigidez à flexão (módulo de Young × momento de inércia) | N·m²    |
| `q`     | Carga distribuída uniforme                              | N/m     |
| `L`     | Comprimento da viga                                     | m       |

### Condições de contorno — apoio simples nas duas extremidades

```text
w(0)   = 0    → deslocamento nulo na extremidade esquerda
w(L)   = 0    → deslocamento nulo na extremidade direita
w''(0) = 0    → momento fletor nulo na extremidade esquerda
w''(L) = 0    → momento fletor nulo na extremidade direita
```

### Solução analítica exata

Para `EI = 1`, `q = 1` e `L = 1`:

```text
w(x) = (q / 24EI) · x · (L³ − 2Lx² + x³)
```

Esta solução serve como referência de validação para ambas as abordagens.

---

## Abordagens implementadas

### Problema direto — `pinn_euler_bernoulli.py`

**Entrada:** parâmetros físicos conhecidos (`EI`, `q`, `L`) e condições de contorno.
**Saída:** função de deslocamento `w(x)` aprendida pela rede.

A rede neural aprende `w(x)` minimizando uma função de perda composta por dois termos:

```text
Loss = L_pde + 10 · L_bc
```

```text
L_pde = média[(EI · ŵ'''' − q)²]                 sobre 200 pontos de colocação
L_bc  = ŵ(0)² + ŵ(L)² + ŵ''(0)² + ŵ''(L)²       nos 4 pontos de borda
```

**Resultado:** erro relativo L² ≈ 3% em 15 000 épocas na CPU.

---

### Problema inverso — `pinn_inverso_EI.py`

**Entrada:** medições ruidosas de `w(x)` em posições esparsas.
**Saída:** função `w(x)` e o parâmetro `EI` identificado simultaneamente.

Nesta abordagem, `EI` deixa de ser uma constante conhecida e passa a ser um **parâmetro treinável** otimizado junto com os pesos da rede:

```python
self.log_EI = nn.Parameter(torch.log(torch.tensor(0.5)))  # chute inicial: EI = 0.5
```

A função de perda ganha um terceiro termo: o ajuste aos dados medidos.

```text
Loss = L_pde + 10 · L_bc + 100 · L_data
```

com

```text
L_data = média[(ŵ(xᵢ) − w_medido(xᵢ))²]
```

avaliado sobre os `N` pontos de sensor.

**Resultado com 20 sensores e 2% de ruído:** erro na identificação de `EI` < 0,4% em 20 000 épocas.

---

## Arquitetura da rede neural

Ambas as abordagens usam a mesma arquitetura MLP:

```text
Entrada: x ∈ ℝ (posição ao longo da viga)
     ↓
Linear(1 → 64) + Tanh
Linear(64 → 64) + Tanh
Linear(64 → 64) + Tanh
Linear(64 → 64) + Tanh
Linear(64 → 1)
     ↓
Saída: ŵ(x) ∈ ℝ (deslocamento previsto)
```

## Derivadas automáticas

O cálculo do resíduo `EI · ŵ'''' − q` é feito inteiramente via autograd do PyTorch.

```python
def nth_derivative(f, x, n):
    df = f

    for _ in range(n):
        df, = torch.autograd.grad(
            df,
            x,
            grad_outputs=torch.ones_like(df),
            create_graph=True,
            retain_graph=True
        )

    return df
```

---

## SDOF — `pinn_sdof.py`

Sistema de 1 grau de liberdade (*Single Degree of Freedom*) com amortecimento viscoso, baseado na Seção 3.1 e Equações 20–21 de Rosafalco et al. (2023).

### Equação do movimento

```text
ẍ(t) + 2ζω·ẋ(t) + ω²·x(t) = 0
```

### Solução analítica (sistema subamortecido, ζ < 1) — Eq. 20

```text
x(t) = e^(−ζωt) · [x₀·cos(ωd·t) +
                   ((ẋ₀ + ζωx₀)/ωd)·sin(ωd·t)]

onde

ωd = ω·√(1 − ζ²)
```

| Símbolo | Descrição                     | Unidade |
| ------- | ----------------------------- | ------- |
| `x(t)`  | Deslocamento do sistema       | m       |
| `ω`     | Frequência natural            | rad/s   |
| `ζ`     | Razão de amortecimento        | —       |
| `ωd`    | Frequência natural amortecida | rad/s   |

### Estratégia (conforme o artigo)

```text
Entrada  I = [x(tᵢ); ẍ(tᵢ)],        i = 1..N
Saída    O = [x(tᵢ); ẍ(tᵢ); ω; ζ],  i > N
```

A rede é treinada apenas sobre o intervalo `[0, T_train]` e então usada para predizer o comportamento além desse período, explorando a restrição física aprendida durante o treinamento.

### Função de perda — Eq. 21

```text
L = (1/N) Σ[(x*ᵢ − x̂ᵢ)² + (ẍ*ᵢ − ẍ̂ᵢ)²]
  + λ · (1/N) Σ[ẍ̂ᵢ + 2ζ̂ω̂·ẋ̂ᵢ + ω̂²·x̂ᵢ]²
  + λ · [x₀* − x̂(0)]²
```

onde:

* Primeiro termo: ajuste aos dados experimentais;
* Segundo termo: resíduo físico da EDO;
* Terceiro termo: condição inicial.

O parâmetro `λ` controla o peso da física:

| Modo | λ | Comportamento                                                        |
| ---- | - | -------------------------------------------------------------------- |
| PINN | 1 | Física + dados; identifica `ω` e `ζ` e extrapola corretamente        |
| ANN  | 0 | Apenas dados; falha na extrapolação e na identificação de parâmetros |

`ω` e `ζ` são parametrizados para garantir positividade e subamortecimento durante todo o treinamento:

```python
self.omega = softplus(raw_omega)  # ω > 0
self.zeta  = sigmoid(raw_zeta)    # 0 < ζ < 1
```

**Resultado com 2 s de dados (de um total de 5 s) e 2% de ruído:** erro na identificação de `ω` < 50% e erro em `ζ` > 900% após 20 000 épocas.

---

## Referências

* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
* Wang, S., Yu, X., & Perdikaris, P. (2022). *When and why PINNs fail to train: A neural tangent kernel perspective*. Journal of Computational Physics, 449, 110768.
* Repositório original dos autores: https://github.com/maziarraissi/PINNs
* Documentação PyTorch autograd: https://pytorch.org/docs/stable/autograd.html
