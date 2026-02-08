"""
Demo script showing how to use the MLP PINN architecture.

This script demonstrates:
1. Creating an MLP model with configurable architecture
2. Forward pass and activation extraction
3. Computing PDE residuals
4. Training step with loss decomposition
"""

import torch
from src.models import MLP

# Set random seed for reproducibility
torch.manual_seed(42)

print("=" * 70)
print("MLP PINN Architecture Demo")
print("=" * 70)

# 1. Create MLP model
print("\n1. Creating MLP model...")
model = MLP(
    input_dim=2,
    hidden_dims=[64, 64, 64],
    output_dim=1,
    activation='tanh'
)
print(model)

# 2. Forward pass and activation extraction
print("\n2. Forward pass with activation extraction...")
x = torch.randn(100, 2)
u = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {u.shape}")

activations = model.get_activations()
print(f"   Extracted activations for {len(activations)} layers:")
for layer_name, act in activations.items():
    print(f"   - {layer_name}: {act.shape}")

# 3. Test different activation functions
print("\n3. Testing different activation functions...")
for activation in ['tanh', 'relu', 'gelu', 'sin']:
    model_act = MLP(input_dim=2, hidden_dims=[50], output_dim=1, activation=activation)
    u_act = model_act(x)
    act_values = model_act.get_activations()['layer_0']
    print(f"   {activation:6s}: output range [{u_act.min():.3f}, {u_act.max():.3f}], "
          f"activation range [{act_values.min():.3f}, {act_values.max():.3f}]")

# 4. Compute PDE residual (Poisson equation example)
print("\n4. Computing PDE residual for Poisson equation...")
x_pde = torch.randn(100, 2, requires_grad=True)

def poisson_pde(u, x, du_dx, d2u_dx2):
    """Poisson equation: ∇²u = f(x) with f(x) = -2π²sin(πx)sin(πy)"""
    f = -2 * (torch.pi**2) * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
    return d2u_dx2 - f

model_poisson = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
residual = model_poisson.compute_pde_residual(x_pde, pde_fn=poisson_pde)
print(f"   Residual shape: {residual.shape}")
print(f"   Mean absolute residual: {residual.abs().mean().item():.6f}")

# 5. Training step demonstration
print("\n5. Training step with loss decomposition...")
x_interior = torch.randn(200, 2, requires_grad=True)
x_boundary = torch.randn(80, 2)
u_boundary = torch.zeros(80, 1)  # Dirichlet BC: u = 0 on boundary

losses = model_poisson.train_step(
    x_interior=x_interior,
    x_boundary=x_boundary,
    u_boundary=u_boundary,
    pde_fn=poisson_pde,
    weights={'pde': 1.0, 'bc': 10.0, 'ic': 0.0}  # Higher weight on boundary
)

print("   Loss components:")
for loss_name, loss_value in losses.items():
    print(f"   - {loss_name:12s}: {loss_value.item():.6f}")

# 6. Quick training demo
print("\n6. Quick training demonstration (10 iterations)...")
model_train = MLP(input_dim=2, hidden_dims=[50, 50], output_dim=1)
optimizer = torch.optim.Adam(model_train.parameters(), lr=1e-3)

print("   Iter | Total Loss | PDE Loss  | BC Loss")
print("   " + "-" * 45)

for i in range(10):
    optimizer.zero_grad()

    # Generate random collocation points
    x_int = torch.randn(200, 2, requires_grad=True)
    x_bc = torch.randn(80, 2)
    u_bc = torch.zeros(80, 1)

    # Compute losses
    losses = model_train.train_step(x_int, x_bc, u_bc, pde_fn=poisson_pde)

    # Backpropagate
    losses['loss_total'].backward()
    optimizer.step()

    if i % 2 == 0:
        print(f"   {i:4d} | {losses['loss_total'].item():10.6f} | "
              f"{losses['loss_pde'].item():9.6f} | {losses['loss_bc'].item():9.6f}")

print("\n" + "=" * 70)
print("Demo completed successfully!")
print("=" * 70)
