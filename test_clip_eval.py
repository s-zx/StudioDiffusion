from pathlib import Path
from evaluation.clip_alignment import CLIPPlatformAlignment

# Initialize evaluator (use cpu for Mac)
evaluator = CLIPPlatformAlignment(device="cpu")

# Build reference embeddings
platform_dirs = {
    "shopify": Path("data/platform_sets/shopify"),
    "etsy": Path("data/platform_sets/etsy"),
    "ebay": Path("data/platform_sets/ebay"),
}

evaluator.build_reference_embeddings(platform_dirs)

# TEMP: use shopify images as fake "generated" images just to test pipeline
gen_images = list(Path("data/platform_sets/shopify").glob("*.jpg"))[:10]

results = evaluator.evaluate(gen_images, target_platform="shopify")
print(results)
