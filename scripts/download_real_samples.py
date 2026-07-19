import os
import urllib.request
from PIL import Image
import glob

# URLs of real industrial/metal images from public domain sources (Wikimedia Commons)
IMAGES = {
    "real_normal.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Stainless_steel_surface.jpg",
    "real_light.jpg": "https://upload.wikimedia.org/wikipedia/commons/9/91/Brushed_aluminum_surface.jpg",
    "real_scratch.jpg": "https://upload.wikimedia.org/wikipedia/commons/0/07/Scratched_metal_texture_%284824318731%29.jpg",
    "real_corrosion.jpg": "https://upload.wikimedia.org/wikipedia/commons/2/23/Rusted_metal_texture.jpg",
    "real_stain.jpg": "https://upload.wikimedia.org/wikipedia/commons/6/6c/Rusty_Metal_Texture_%28205244535%29.jpg",
    "real_missing.jpg": "https://upload.wikimedia.org/wikipedia/commons/5/5c/Bullet_hole_in_metal.jpg"
}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. Delete old synthetic benchmark images
    old_bench_images = glob.glob(os.path.join(raw_dir, "bench_*.jpg"))
    for f in old_bench_images:
        try:
            os.remove(f)
            print(f"Deleted old synthetic image: {os.path.basename(f)}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")
            
    # 2. Download and resize new real images
    for filename, url in IMAGES.items():
        filepath = os.path.join(raw_dir, filename)
        print(f"Downloading {filename}...")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            
            # Process image (resize to 300x300, convert to RGB)
            with Image.open(filepath) as img:
                img = img.convert('RGB')
                # Center crop to aspect ratio then resize
                width, height = img.size
                min_dim = min(width, height)
                left = (width - min_dim)/2
                top = (height - min_dim)/2
                right = (width + min_dim)/2
                bottom = (height + min_dim)/2
                img = img.crop((left, top, right, bottom))
                img = img.resize((300, 300), Image.Resampling.LANCZOS)
                img.save(filepath, format="JPEG", quality=90)
                
            print(f"Successfully saved processed {filename}")
        except Exception as e:
            print(f"Failed to download/process {filename}: {e}")

if __name__ == "__main__":
    main()
