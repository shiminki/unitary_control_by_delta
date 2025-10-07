import torch
import numpy as np

import sys

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from model.unitary_control_transformer import UnitaryControlTransformer
from model.unitary_control_trainer import UniversalModelTrainer

from train.train import *

def euler_yxy_from_rotation_vector(rotation_vector: torch.Tensor,
                                eps: float = 1e-12) -> torch.Tensor:
    """
    Vectorised Y-X-Y Euler decomposition.
    Args
    ----
        rotation_vector : (B,4) tensor (n_x, n_y, n_z, θ)
    Returns
    -------
        (B,3) tensor (α, β, γ) such that
            exp(-i θ/2 n·σ) = R_y(α) · R_x(β) · R_y(γ)
    """
    n, theta = rotation_vector[..., :3], rotation_vector[..., 3]
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(eps)       # normalise axis

    s, c = torch.sin(theta / 2), torch.cos(theta / 2)          # sin, cos θ/2
    w, x, y, z = c, n[..., 0] * s, n[..., 1] * s, n[..., 2] * s

    # ----- regular branch -------------------------------------------------- #
    beta = torch.acos((1.0 - 2.0 * (x**2 + z**2)).clamp(-1.0 + eps, 1.0 - eps))
    sin_beta = beta.sin()

    alpha_reg = torch.atan2(x * y + z * w, w * x - y * z)
    gamma_reg = torch.atan2(x * y - z * w, w * x + y * z)

    # ----- gimbal-lock handling ------------------------------------------- #
    tol = 1e-6
    mask_reg  = sin_beta.abs() > tol          # "normal" points
    mask_beta0 = ~mask_reg & (beta < 0.5)     # β ≈ 0         → Y-only
    mask_betapi = ~mask_reg & ~mask_beta0     # β ≈ π         → X/Z

    alpha = torch.where(mask_reg, alpha_reg, torch.zeros_like(alpha_reg))
    gamma = torch.where(mask_reg, gamma_reg, torch.zeros_like(gamma_reg))

    # β ≈ 0  (rotation about Y)
    # When sin(β/2) ≈ 0, we have x ≈ z ≈ 0, so rotation is about Y
    # α + γ = 2*atan2(y, w), we set γ=0 conventionally
    alpha = torch.where(mask_beta0, 2.0 * torch.atan2(y, w), alpha)
    gamma = torch.where(mask_beta0, torch.zeros_like(gamma), gamma)

    # β ≈ π  (rotation about X or Z)
    # When sin(β/2) ≈ 1, we have w ≈ y ≈ 0
    # α - γ = 2*atan2(z, x), we set α=0 conventionally  
    alpha = torch.where(mask_betapi, torch.zeros_like(alpha), alpha)
    gamma = torch.where(mask_betapi, 2.0 * torch.atan2(z, x), gamma)

    return torch.stack((alpha, beta, gamma), dim=-1)


# Helper functions to construct rotation matrices
def pauli_matrices():
    """Returns Pauli matrices X, Y, Z"""
    X = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64)
    Y = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex64)
    Z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64)
    return X, Y, Z


def rotation_from_axis_angle(n_x, n_y, n_z, theta):
    """Compute U = exp(-i θ/2 n·σ)"""
    X, Y, Z = pauli_matrices()
    n_sigma = n_x * X + n_y * Y + n_z * Z
    
    # Matrix exponential: exp(-i θ/2 n·σ)
    U = torch.matrix_exp(-1j * (theta / 2) * n_sigma)
    return U


def rotation_y(angle):
    """R_y(angle) = exp(-i angle/2 Y)"""
    c = torch.cos(angle / 2)
    s = torch.sin(angle / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)


def rotation_x(angle):
    """R_x(angle) = exp(-i angle/2 X)"""
    c = torch.cos(angle / 2)
    s = torch.sin(angle / 2)
    return torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=torch.complex64)


def compare_unitaries(U1, U2, tol=1e-4):
    """Compare two unitaries up to global phase"""
    # Check if U1 ≈ e^(iφ) U2 for some phase φ
    U_diff = U1.conj().T @ U2
    diff = torch.trace(U_diff).abs() / 2 - 1  # |Tr(U1†U2)|/2 - 1
    return diff < tol, diff.item()


# Test Cases
print("=" * 70)
print("TESTING EULER Y-X-Y DECOMPOSITION")
print("=" * 70)

# Test 1: Identity (theta = 0)
print("\n[Test 1] Identity rotation (θ=0)")
rotation_vec = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # any axis, θ=0
euler = euler_yxy_from_rotation_vector(rotation_vec)
print(f"Input: n=[1,0,0], θ=0")
print(f"Euler angles (α,β,γ): {euler[0].numpy()}")

U_original = rotation_from_axis_angle(1.0, 0.0, 0.0, 0.0)
U_reconstructed = rotation_y(euler[0, 0]) @ rotation_x(euler[0, 1]) @ rotation_y(euler[0, 2])
match, diff = compare_unitaries(U_original, U_reconstructed)
print(f"Match: {match}, Max diff: {diff:.2e}")


# Test 2: Pure Y rotation
print("\n[Test 2] Pure Y-axis rotation (θ=π/3)")
theta = np.pi / 3
rotation_vec = torch.tensor([[0.0, 1.0, 0.0, theta]])
euler = euler_yxy_from_rotation_vector(rotation_vec)
print(f"Input: n=[0,1,0], θ={theta:.4f}")
print(f"Euler angles (α,β,γ): {euler[0].numpy()}")

U_original = rotation_from_axis_angle(0.0, 1.0, 0.0, theta)
U_reconstructed = rotation_y(euler[0, 0]) @ rotation_x(euler[0, 1]) @ rotation_y(euler[0, 2])
match, diff = compare_unitaries(U_original, U_reconstructed)
print(f"Match: {match}, Max diff: {diff:.2e}")


# Test 3: Pure X rotation
print("\n[Test 3] Pure X-axis rotation (θ=π/4)")
theta = np.pi / 4
rotation_vec = torch.tensor([[1.0, 0.0, 0.0, theta]])
euler = euler_yxy_from_rotation_vector(rotation_vec)
print(f"Input: n=[1,0,0], θ={theta:.4f}")
print(f"Euler angles (α,β,γ): {euler[0].numpy()}")

U_original = rotation_from_axis_angle(1.0, 0.0, 0.0, theta)
U_reconstructed = rotation_y(euler[0, 0]) @ rotation_x(euler[0, 1]) @ rotation_y(euler[0, 2])
match, diff = compare_unitaries(U_original, U_reconstructed)
print(f"Match: {match}, Max diff: {diff:.2e}")


# Test 4: Pure Z rotation
print("\n[Test 4] Pure Z-axis rotation (θ=π/6)")
theta = np.pi / 6
rotation_vec = torch.tensor([[0.0, 0.0, 1.0, theta]])
euler = euler_yxy_from_rotation_vector(rotation_vec)
print(f"Input: n=[0,0,1], θ={theta:.4f}")
print(f"Euler angles (α,β,γ): {euler[0].numpy()}")

U_original = rotation_from_axis_angle(0.0, 0.0, 1.0, theta)
U_reconstructed = rotation_y(euler[0, 0]) @ rotation_x(euler[0, 1]) @ rotation_y(euler[0, 2])
match, diff = compare_unitaries(U_original, U_reconstructed)
print(f"Match: {match}, Max diff: {diff:.2e}")


# Test 5: General rotation
print("\n[Test 5] General rotation (arbitrary axis)")
n_x, n_y, n_z = 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)
theta = np.pi / 2
rotation_vec = torch.tensor([[n_x, n_y, n_z, theta]])
euler = euler_yxy_from_rotation_vector(rotation_vec)
print(f"Input: n=[{n_x:.4f},{n_y:.4f},{n_z:.4f}], θ={theta:.4f}")
print(f"Euler angles (α,β,γ): {euler[0].numpy()}")

U_original = rotation_from_axis_angle(n_x, n_y, n_z, theta)
U_reconstructed = rotation_y(euler[0, 0]) @ rotation_x(euler[0, 1]) @ rotation_y(euler[0, 2])
match, diff = compare_unitaries(U_original, U_reconstructed)
print(f"Match: {match}, Max diff: {diff:.2e}")


# Test 6: Gimbal lock case (β ≈ π)
print("\n[Test 6] Gimbal lock: rotation by π about X-axis")
theta = np.pi
rotation_vec = torch.tensor([[1.0, 0.0, 0.0, theta]])
euler = euler_yxy_from_rotation_vector(rotation_vec)
print(f"Input: n=[1,0,0], θ={theta:.4f}")
print(f"Euler angles (α,β,γ): {euler[0].numpy()}")

U_original = rotation_from_axis_angle(1.0, 0.0, 0.0, theta)
U_reconstructed = rotation_y(euler[0, 0]) @ rotation_x(euler[0, 1]) @ rotation_y(euler[0, 2])
match, diff = compare_unitaries(U_original, U_reconstructed)
print(f"Match: {match}, Max diff: {diff:.2e}")


# Test 7: Batch processing
print("\n[Test 7] Batch processing (5 rotations)")
rotation_vecs = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],           # identity
    [0.0, 1.0, 0.0, np.pi/2],       # Y rotation
    [1.0, 0.0, 0.0, np.pi/4],       # X rotation
    [0.0, 0.0, 1.0, np.pi/3],       # Z rotation
    [1/np.sqrt(2), 1/np.sqrt(2), 0.0, np.pi/6]  # XY plane rotation
])
eulers = euler_yxy_from_rotation_vector(rotation_vecs)
print(f"Batch shape: {eulers.shape}")

all_match = True
for i in range(len(rotation_vecs)):
    rv = rotation_vecs[i]
    euler = eulers[i]
    U_original = rotation_from_axis_angle(rv[0], rv[1], rv[2], rv[3])
    U_reconstructed = rotation_y(euler[0]) @ rotation_x(euler[1]) @ rotation_y(euler[2])
    match, diff = compare_unitaries(U_original, U_reconstructed)
    all_match = all_match and match
    print(f"  Rotation {i+1}: Match={match}, Diff={diff:.2e}")

print(f"\nAll batch tests passed: {all_match}")

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)



print("=" * 70)
print("SU(2) DATALOADER TEST")
print("=" * 70)

# Generate dataset
dataset_size = 30000
max_N = 4
batch_size = 800

print(f"\nGenerating dataset...")
print(f"  Dataset size: {dataset_size}")
print(f"  Max N: {max_N}")
print(f"  Batch size: {batch_size}")

dataset = build_SU2_dataset(dataset_size=dataset_size, max_N=max_N)
print(f"  Total data entries: {len(dataset)}")

# Create dataloader
dataloader = SU2DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Print statistics
print("\nDataLoader Statistics:")
stats = dataloader.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

# Expected number of batches per length
print("\nExpected batches per sequence length:")
for n in range(1, max_N + 1):
    count = dataset_size // max_N  # 7500 for dataset_size=30000, max_N=4
    num_full = count // batch_size
    remaining = count % batch_size
    total_batches = num_full + (1 if remaining > 0 else 0)
    print(f"  n={n}: {num_full} full batches ({batch_size} samples) + "
            f"{1 if remaining > 0 else 0} trailing batch ({remaining} samples) = {total_batches} total")

# Iterate through dataloader
print("\nIterating through batches:")
batch_shapes = defaultdict(list)

for i, batch in enumerate(dataloader):
    n = batch.shape[1]
    batch_size_actual = batch.shape[0]
    batch_shapes[n].append(batch_size_actual)
    
    # Validate shape
    assert batch.shape[-1] == 6, f"Expected last dim to be 6, got {batch.shape[-1]}"

# Print batch distribution
print("\nBatch distribution by sequence length:")
total_batches = 0
for n in sorted(batch_shapes.keys()):
    sizes = batch_shapes[n]
    print(f"  n={n}: {len(sizes)} batches with sizes {sizes}")
    total_batches += len(sizes)

print(f"\nTotal batches yielded: {total_batches}")
print(f"Expected total batches: {len(dataloader)}")
assert total_batches == len(dataloader), "Mismatch in batch count!"

print("\n✓ All tests passed!")
print("=" * 70)

print(f"Example batch data (first batch):   {next(iter(dataloader)).shape}")




from model.unitary_control_transformer import UnitaryControlTransformer

model = UnitaryControlTransformer(
    num_qubits=1, pulse_space={'omega_sys': [0.0, 1.0], 'phi_sys': [-3.15, 3.15], "tau": [0.1, 0.5]},
    max_pulses=100
)

X = torch.randn(10, 4, 6)

model(X)