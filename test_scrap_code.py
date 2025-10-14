import torch
import numpy as np

import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from model.UnitaryControlTransformer import UnitaryControlTransformer
from model.UniversalModelTrainer import UniversalModelTrainer

from train import *

from util.dataset.delta_control_SU2_dataset import build_SU2_dataset
from util.dataset.SU2DataLoader import SU2DataLoader
from util.fidelity.fidelity import fidelity_single_qubit_control



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




from model.UnitaryControlTransformer import UnitaryControlTransformer

model = UnitaryControlTransformer(
    num_qubits=1, pulse_space={'omega_sys': [0.0, 1.0], 'phi_sys': [-3.15, 3.15], "tau": [0.1, 0.5]},
    max_pulses=100
)

X = torch.randn(10, 4, 6)

model(X)





#!/usr/bin/env python
"""
Debug script to identify CUDA memory access issues in the training pipeline.
Run this before the main training to validate the setup.
"""

import torch
import sys
import os
import traceback

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def test_unitary_generator():
    """Test the unitary generator with small batches."""
    print("\n" + "="*60)
    print("Testing Unitary Generator for Two Qubits")
    print("="*60)
    
    from util.unitary_generator.unitary_generators import batched_unitary_generator_two_qubits_entangled
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test with small batch
    B, L = 2, 10  # batch size, sequence length
    
    # Create test data
    pulses = torch.randn(B, L, 5, device=device)
    errors = torch.randn(B, 4, device=device)
    
    print(f"Pulses shape: {pulses.shape}")
    print(f"Errors shape: {errors.shape}")
    
    try:
        # Test unitary generation
        U_out = batched_unitary_generator_two_qubits_entangled(pulses, errors)
        print(f"Output unitary shape: {U_out.shape}")
        print(f"Output dtype: {U_out.dtype}")
        print(f"Output device: {U_out.device}")
        
        # Verify unitary properties
        U_dagger = U_out.conj().transpose(-2, -1)
        I_check = torch.bmm(U_out, U_dagger)
        identity_error = torch.norm(I_check - torch.eye(4, device=device), dim=(-2, -1)).max()
        print(f"Max deviation from unitarity: {identity_error.item():.6e}")
        
        print("✓ Unitary generator test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Unitary generator test failed!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_model_forward():
    """Test the model forward pass."""
    print("\n" + "="*60)
    print("Testing Model Forward Pass")
    print("="*60)
    
    from model.UnitaryControlTransformer import UnitaryControlTransformer
    import json
    
    # Load model config
    config_path = "train/train_with_entanglement/model_config.json"
    with open(config_path, "r") as f:
        model_params = json.load(f)
    
    # Convert pulse_space values to tuples
    if "pulse_space" in model_params:
        for k, v in model_params["pulse_space"].items():
            model_params["pulse_space"][k] = tuple(v)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnitaryControlTransformer(**model_params).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test input
    batch_size = 2
    n = 3  # number of rotation vectors
    test_input = torch.randn(batch_size, n, 6, device=device)
    
    print(f"Input shape: {test_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        print("✓ Model forward pass test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model forward pass test failed!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation with fidelity."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)
    
    from util.fidelity.fidelity import fidelity_entangled_control
    from util.loss_function.loss_function import sharp_loss
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test unitaries
    B = 4
    U_out = torch.eye(4, dtype=torch.cfloat, device=device).unsqueeze(0).expand(B, -1, -1)
    U_target = torch.eye(2, dtype=torch.cfloat, device=device).unsqueeze(0).expand(B, -1, -1)
    
    print(f"U_out shape: {U_out.shape}")
    print(f"U_target shape: {U_target.shape}")
    
    try:
        # Test fidelity computation
        fidelity = fidelity_entangled_control(U_out, U_target)
        print(f"Fidelity shape: {fidelity.shape}")
        print(f"Fidelity values: {fidelity}")
        
        # Test loss computation
        loss = sharp_loss(U_out, U_target, fidelity_fn=fidelity_entangled_control)
        print(f"Loss: {loss.item():.6f}")
        
        print("✓ Loss computation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Loss computation test failed!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("Testing Single Training Step")
    print("="*60)
    
    from model.UnitaryControlTransformer import UnitaryControlTransformer
    from model.UniversalModelTrainer import UniversalModelTrainer
    from util.dataset.delta_control_SU2_dataset import build_SU2_dataset
    from util.fidelity.fidelity import fidelity_entangled_control
    from util.loss_function.loss_function import sharp_loss
    from util.unitary_generator.unitary_generators import batched_unitary_generator_two_qubits_entangled
    from util.error_sampler.error_sampler import two_qubit_error_sampler
    import json
    
    # Load model config
    config_path = "train/train_with_entanglement/model_config.json"
    with open(config_path, "r") as f:
        model_params = json.load(f)
    
    # Convert pulse_space values to tuples
    if "pulse_space" in model_params:
        for k, v in model_params["pulse_space"].items():
            model_params["pulse_space"][k] = tuple(v)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnitaryControlTransformer(**model_params)
    
    # Create trainer with small monte carlo for testing
    trainer = UniversalModelTrainer(
        model=model,
        unitary_generator=batched_unitary_generator_two_qubits_entangled,
        error_sampler=two_qubit_error_sampler,
        fidelity_fn=fidelity_entangled_control,
        loss_fn=sharp_loss,
        device=str(device),
        monte_carlo=10  # Small for testing
    )
    
    # Create small test batch
    test_dataset = build_SU2_dataset(dataset_size=4, max_N=2, max_delta=1.5)
    test_batch = torch.stack(test_dataset[:2])  # Take first 2 samples
    
    print(f"Test batch shape: {test_batch.shape}")
    
    try:
        # Run one training step
        loss, fidelity = trainer.train_epoch(test_batch)
        print(f"Loss: {loss:.6f}")
        print(f"Fidelity: {fidelity:.6f}")
        
        print("✓ Training step test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training step test failed!")
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Additional debugging
        if "illegal memory access" in str(e).lower():
            print("\nCUDA Memory Access Error Detected!")
            print("This typically happens due to:")
            print("1. Tensor shape mismatches")
            print("2. Operations on tensors from different devices")
            print("3. In-place operations on tensors used in autograd")
            
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("CUDA Debugging Script for Quantum Control Training")
    print("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA Available: No (using CPU)")
    
    # Set CUDA debugging flags
    if torch.cuda.is_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.cuda.synchronize()
        print("\nCUDA_LAUNCH_BLOCKING enabled for better error messages")
    
    # Run tests
    tests = [
        test_unitary_generator,
        test_model_forward,
        test_loss_computation,
        test_training_step
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            if not result:
                print(f"\nStopping tests due to failure in {test.__name__}")
                break
        except Exception as e:
            print(f"\nUnexpected error in {test.__name__}: {e}")
            results.append(False)
            break
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The training should work.")
    else:
        print("\n✗ Some tests failed. Please fix the issues before training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)