

import numpy as np
import joblib
import rasterio
from rasterio.windows import Window
from rasterio.profiles import DefaultGTiffProfile
import os
import time
import psutil

band_folder = r'D:\personalprject'
output_folder = r'D:\predictions'
model_path = r'D:\personalprject\rabi_model_enhanced.pkl'  # Change for Rabi
season = 'nov'  # Change to 'nov' for Rabi
# =================

os.makedirs(output_folder, exist_ok=True)

# Auto-detect patch size based on RAM
ram_gb = psutil.virtual_memory().total / (1024**3)
if ram_gb >= 16:
    patch_size = 2000
elif ram_gb >= 8:
    patch_size = 1500
else:
    patch_size = 1000


print(f"Loading model for {season.upper()}...")
model = joblib.load(model_path)
print(f"Model expects {model.n_features_in_} features")
b02_src = rasterio.open(os.path.join(band_folder, f'{season}_B02.jp2'))
b03_src = rasterio.open(os.path.join(band_folder, f'{season}_B03.jp2'))
b04_src = rasterio.open(os.path.join(band_folder, f'{season}_B04.jp2'))
b08_src = rasterio.open(os.path.join(band_folder, f'{season}_B08.jp2'))
h = b02_src.height
w = b02_src.width
print(f"Image size: {h} x {w} = {h*w:,} pixels")

n_patches_h = (h + patch_size - 1) // patch_size
n_patches_w = (w + patch_size - 1) // patch_size
total_patches = n_patches_h * n_patches_w
print(f"Processing {total_patches} patches...")
profile = b02_src.profile
profile.update(
    driver='GTiff',
    count=1,
    dtype='float32',
    compress='lzw',
    tiled=True,
    bigtiff='YES' if h*w > 10000*10000 else 'NO'
)

with rasterio.open(
    os.path.join(output_folder, f'predicted_{season}_to_sep.tif'),
    'w',
    **profile
) as dst:
    
    start_time = time.time()
    patch_count = 0
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate window coordinates
            y_off = i * patch_size
            x_off = j * patch_size
            y_size = min(patch_size, h - y_off)
            x_size = min(patch_size, w - x_off)
            
            window = Window(x_off, y_off, x_size, y_size)
            patch_count += 1
            
            # ETA calculation
            elapsed = time.time() - start_time
            if patch_count > 1:
                est_total = (elapsed / (patch_count-1)) * total_patches
                remaining = est_total - elapsed
            else:
                remaining = 0
            
            print(f"   Patch {patch_count}/{total_patches} "
                  f"({y_size}x{x_size}) - "
                  f"ETA: {remaining/60:.1f} min", end='\r')
            
            # Read bands for this window
            b02 = b02_src.read(1, window=window).astype(np.float32)
            b03 = b03_src.read(1, window=window).astype(np.float32)
            b04 = b04_src.read(1, window=window).astype(np.float32)
            b08 = b08_src.read(1, window=window).astype(np.float32)
            
            # Small epsilon to avoid division by zero
            eps = 1e-10
            
            # Calculate ALL 12 indices
            # 1. NDVI
            ndvi = (b08 - b04) / (b08 + b04 + eps)
            
            # 2. Ratio84
            ratio84 = b08 / (b04 + eps)
            
            # 3. Brightness
            brightness = (b02 + b03 + b04) / 3
            
            # 4. Diff84
            diff84 = b08 - b04
            
            # 5. EVI
            evi = 2.5 * ((b08 - b04) / (b08 + 6*b04 - 7.5*b02 + 1 + eps))
            
            # 6. SAVI
            L = 0.5
            savi = ((b08 - b04) / (b08 + b04 + L + eps)) * (1 + L)
            
            # 7. GNDVI
            gndvi = (b08 - b03) / (b08 + b03 + eps)
            
            # 8. VARI
            vari = (b03 - b04) / (b03 + b04 - b02 + eps)
            
            # 9. NDWI
            ndwi = (b03 - b08) / (b03 + b08 + eps)
            
            # 10. CI
            ci = (b08 / (b03 + eps)) - 1
            
            # 11. NDVIre
            ndvire = (b08 - b04) / (b08 + b04 + eps)
            
            # 12. MSAVI
            msavi = (2 * b08 + 1 - np.sqrt((2 * b08 + 1)**2 - 8 * (b08 - b04) + eps)) / 2
            
            # Stack ALL 16 features in correct order
            features = np.stack([
                b02, b03, b04, b08,
                ndvi, ratio84, brightness, diff84,
                evi, savi, gndvi, vari,
                ndwi, ci, ndvire, msavi
            ], axis=-1)
            
            # Reshape for prediction
            h_patch, w_patch = b02.shape
            features_flat = features.reshape(-1, 16)
            
            # Predict
            pred_flat = model.predict(features_flat)
            pred_2d = pred_flat.reshape(h_patch, w_patch)
            
            # Write to output
            dst.write(pred_2d, 1, window=window)


total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
