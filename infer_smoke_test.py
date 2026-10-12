import collections
import collections.abc

# Compatibility patch for older libraries running on Python 3.10+
for abc_name in ['Iterable', 'Mapping', 'MutableMapping', 'Sequence', 'Callable', 'Set']:
    if not hasattr(collections, abc_name):
        setattr(collections, abc_name, getattr(collections.abc, abc_name))

# Compatibility patch for MOABB dataset naming changes
import moabb.datasets as mbd
if not hasattr(mbd, 'BNCI2014001') and hasattr(mbd, 'BNCI2014_001'):
    mbd.BNCI2014001 = mbd.BNCI2014_001

import torch
from dn3_ext import BENDRClassification
def test_bendr_inference():
    print("Initializing BENDR inference smoke test...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths to your downloaded v0.1-alpha checkpoint weights
    encoder_checkpoint = "path/to/encoder_weights.pt"
    contextualizer_checkpoint = "path/to/contextualizer_weights.pt"

    # Instantiate model structure matching P300 / default dimensions
    model = BENDRClassification(
        targets=2, 
        samples=1000, 
        channels=20, 
        encoder_h=512, 
        contextualizer_hidden=3076
    )

    print("Loading pretrained modules...")
    try:
        model.load_pretrained_modules(
            encoder_file=encoder_checkpoint,
            contextualizer_file=contextualizer_checkpoint,
            strict=False
        )
    except Exception as e:
        print(f"Note on loading (expected if paths are placeholders yet): {e}")

    model.to(device)
    model.eval()

    # Create synthetic EEG batch: (Batch Size, Channels, Time-samples)
    dummy_eeg = torch.randn(2, 20, 1000).to(device)
    print(f"Input EEG tensor shape: {dummy_eeg.shape}")

    # Run forward pass smoke test
    with torch.no_grad():
        output_features = model.features_forward(dummy_eeg)
        print(f"Model forward output features shape: {output_features.shape}")

    print("Smoke test execution completed successfully!")

if __name__ == "__main__":
    test_bendr_inference()
