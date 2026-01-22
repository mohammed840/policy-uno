"""
Export PyTorch model to Core ML format.

Converts trained Q-Network to Core ML .mlpackage for efficient inference on Apple devices.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.qnet_torch import QNetwork, load_model


def export_to_coreml(
    input_path: str,
    output_path: str,
    device: str = 'cpu'
):
    """
    Export PyTorch Q-Network to Core ML format.
    
    Args:
        input_path: Path to PyTorch model (.pt)
        output_path: Path for Core ML output (.mlpackage)
        device: Device for loading model
    """
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools is not installed.")
        print("Install with: pip install coremltools")
        return False
    
    print(f"Loading PyTorch model from: {input_path}")
    
    # Load model
    device = torch.device(device)
    model = load_model(input_path, device)
    model.eval()
    
    print(f"Model loaded: {model.state_size} → {model.hidden_size} → {model.hidden_size} → {model.action_size}")
    
    # Create example input
    example_input = torch.randn(1, model.state_size)
    
    # Trace the model
    print("Tracing model with TorchScript...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to Core ML
    print("Converting to Core ML...")
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="state",
                shape=(1, model.state_size),
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="q_values", dtype=np.float32)
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13
    )
    
    # Add metadata
    mlmodel.author = "Uno RL"
    mlmodel.short_description = "Q-Network for Uno card game"
    mlmodel.version = "1.0"
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Core ML model to: {output_path}")
    mlmodel.save(str(output_path))
    
    print("Export complete!")
    
    # Verify
    print("\nVerifying exported model...")
    try:
        loaded = ct.models.MLModel(str(output_path))
        test_input = {"state": np.random.randn(1, model.state_size).astype(np.float32)}
        output = loaded.predict(test_input)
        print(f"  Input shape: (1, {model.state_size})")
        print(f"  Output shape: {output['q_values'].shape}")
        print("  Verification passed!")
    except Exception as e:
        print(f"  Verification failed: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to Core ML')
    parser.add_argument('--in', dest='input', type=str, required=True,
                       help='Input PyTorch model path (.pt)')
    parser.add_argument('--out', dest='output', type=str, required=True,
                       help='Output Core ML path (.mlpackage)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for loading model')
    
    args = parser.parse_args()
    
    success = export_to_coreml(args.input, args.output, args.device)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
