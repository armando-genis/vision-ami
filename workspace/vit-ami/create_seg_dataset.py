import os
import shutil
from PIL import Image, ImageDraw
import numpy as np

def polygons_to_mask(width, height, polygons, background_id=0):
    """
    Rasterize polygons into a (height, width) mask.
    polygons: list of (class_id, [(x1,y1), (x2,y2), ...]) in *pixel coords*.
    """
    # Start with background
    mask = np.full((height, width), background_id, dtype=np.uint8)

    for class_id, pts in polygons:
        # Create a temporary blank mask
        temp_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(temp_mask)
        # Fill polygon with 'class_id'
        draw.polygon(pts, outline=class_id, fill=class_id)
        
        # Merge into final mask
        temp_np = np.array(temp_mask, dtype=np.uint8)
        # Overwrite wherever temp_np != 0
        mask = np.where(temp_np != 0, temp_np, mask)
    return mask

def create_seg_masks(source_img_dir, source_label_dir, dest_img_dir, dest_mask_dir, background_id=0,class_ids_collector=None):
    """
    Converts polygon .txt annotations into paletted .png masks and copies images.
    
    source_img_dir:   folder with .jpg images
    source_label_dir: folder with .txt polygon annotations
    dest_img_dir:     folder where to copy images
    dest_mask_dir:    folder where to save new .png masks
    background_id:    label ID for background pixels
    """
    # Ensure output directories exist
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    image_files = sorted(os.listdir(source_img_dir))
    for img_name in image_files:
        # 1) Copy the original image to the new dataset
        src_img_path = os.path.join(source_img_dir, img_name)
        dst_img_path = os.path.join(dest_img_dir, img_name)
        shutil.copy2(src_img_path, dst_img_path)  # preserves metadata

        # 2) Read the polygon annotation
        base_name, _ = os.path.splitext(img_name)
        txt_name = base_name + ".txt"
        txt_path = os.path.join(source_label_dir, txt_name)
        
        # 3) If annotation doesn't exist, entire mask = background
        if not os.path.exists(txt_path):
            with Image.open(src_img_path) as img:
                w, h = img.size
            blank_mask = np.full((h, w), background_id, dtype=np.uint8)
            mask_img = Image.fromarray(blank_mask, mode='L')

            # Convert 'L' to 'P' (paletted)
            mask_pal = mask_img.convert('P')
            # Apply a custom color palette
            apply_custom_palette(mask_pal)

            mask_pal.save(os.path.join(dest_mask_dir, base_name + ".png"))
            continue
        
        # 4) Parse polygons
        with Image.open(src_img_path) as img:
            width, height = img.size

        polygons = []  # list of (class_id, [(x1,y1), (x2,y2), ...])
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                if class_ids_collector is not None:
                    class_ids_collector.add(class_id)  # Collect the ID

                xs = coords[0::2]
                ys = coords[1::2]

                # Convert normalized [0,1] => pixel coords
                pxs = [x * width  for x in xs]
                pys = [y * height for y in ys]
                pts = list(zip(pxs, pys))  # e.g. [(100.2, 50.7), (120.9, 58.3), ...]

                polygons.append((class_id, pts))
        
        # 5) Rasterize polygons into mask
        mask_arr = polygons_to_mask(width, height, polygons, background_id=background_id)

        # Convert to PIL grayscale first
        mask_img = Image.fromarray(mask_arr, mode='L')
        
        # 6) Convert to paletted mode and apply custom palette
        mask_pal = mask_img.convert('P')
        apply_custom_palette(mask_pal)

        # 7) Save mask as .png
        mask_filename = base_name + ".png"
        mask_path = os.path.join(dest_mask_dir, mask_filename)
        mask_pal.save(mask_path)

def apply_custom_palette(pil_image, max_classes=10):
    """
    Assign a simple color palette for up to `max_classes` classes in a 'P' mode image.
    Class 0 = black, Class 1 = red, Class 2 = green, Class 3 = blue, etc.
    If you have more than 10 classes, please extend the palette accordingly.
    """
    palette = []
    
    # Example for 10 distinct classes
    # 0 = black
    # 1 = red
    # 2 = green
    # 3 = blue
    # 4 = cyan
    # 5 = magenta
    # 6 = yellow
    # 7 = gray
    # 8 = orange
    # 9 = pink
    colors = [
        (0, 0, 0),       # class 0  => black
        (255, 0, 0),     # class 1  => red
        (0, 255, 0),     # class 2  => green
        (0, 0, 255),     # class 3  => blue
        (0, 255, 255),   # class 4  => cyan
        (255, 0, 255),   # class 5  => magenta
        (255, 255, 0),   # class 6  => yellow
        (128, 128, 128), # class 7  => gray
        (255, 165, 0),   # class 8  => orange
        (255, 192, 203), # class 9  => pink
    ]
    
    # Fill up to max_classes
    for i in range(max_classes):
        if i < len(colors):
            r, g, b = colors[i]
        else:
            # If we run out of predefined colors, use some default or repeat
            r, g, b = (128, 128, 128)
        palette.extend([r, g, b])
    
    # For the remaining 256 - max_classes entries, just fill with black or gray
    palette.extend([0, 0, 0] * (256 - max_classes))
    
    pil_image.putpalette(palette)

def main():
    # Root path to your original data
    data_root = "/workspace/Data"

    # We'll create a new folder "my_seg_dataset" next to it
    output_root = os.path.join(os.path.dirname(data_root), "my_seg_dataset")

    # Subfolders in the new dataset:
    train_img_out = os.path.join(output_root, "train", "images")
    train_mask_out = os.path.join(output_root, "train", "masks")
    val_img_out = os.path.join(output_root, "val", "images")
    val_mask_out = os.path.join(output_root, "val", "masks")

    # Source directories (existing)
    train_images_in = os.path.join(data_root, "train", "images")
    train_labels_in = os.path.join(data_root, "train", "labels")
    val_images_in   = os.path.join(data_root, "val", "images")
    val_labels_in   = os.path.join(data_root, "val", "labels")

    print(f"Creating segmentation dataset at: {output_root}")

    all_class_ids = set()

    # Generate train set
    print("Processing train set...")
    create_seg_masks(
        source_img_dir=train_images_in,
        source_label_dir=train_labels_in,
        dest_img_dir=train_img_out,
        dest_mask_dir=train_mask_out,
        background_id=0,  # set background class ID
        class_ids_collector=all_class_ids,
    )
    
    # Generate val set
    print("Processing val set...")
    create_seg_masks(
        source_img_dir=val_images_in,
        source_label_dir=val_labels_in,
        dest_img_dir=val_img_out,
        dest_mask_dir=val_mask_out,
        background_id=0,
        class_ids_collector=all_class_ids,
    )

    print("Done! New segmentation dataset structure:")
    print(output_root)
    """
    my_seg_dataset/
      train/
        images/
        masks/
      val/
        images/
        masks/
    """

    # Finally, print the classes found
    print("\nAll classes encountered in .txt files:", sorted(all_class_ids))
    print("Number of classes encountered:", len(all_class_ids))

if __name__ == "__main__":
    main()
