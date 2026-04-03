"""
Smoke test: verify imports resolve without errors.
Actual model tests require GPU + downloaded checkpoints.
"""

def test_imports():
    from segmentation import SAM2Extractor, U2NetExtractor, evaluate_masks
    from adapters import IPAdapterSDXL, LoRASDXL
    from evaluation import CLIPPlatformAlignment, DINOv2Fidelity, AestheticScorer, BoundaryPreservation
    from inference import generate_product_image
