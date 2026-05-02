#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

try:
    from src.hybrid_model import HybridNIDS
    print("✓ Import successful")
    m = HybridNIDS()
    print("✓ Model initialized")
    print(f"DNN Loaded: {m.dnn_loaded}")
    print(f"Mode: {m.get_model_status()['modelo_mode']}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
