"""LoveDA submit script for semantic_segmentation_mamba project.

1. Strip Urban_/Rural_ prefix (Codabench GT filenames have no prefix)
2. Model trained with mask 0-6 (already -1 from raw 1-7):
   0=building, 1=road, 2=water, 3=barren, 4=forest, 5=agriculture, 6=background
   Codabench GT raw 1-7, evaluates with gt-1 → same 0-6 mapping
   Predictions and Codabench gt-1 are aligned, no transform needed
3. Ensure single-channel PNG (pixel values 0-6)
4. Package as submission.zip (flat, no nested dirs)

Usage:
    python loveda_submit.py -i pred/loveda_test -o submission.zip
"""
import argparse
import os
import os.path as osp
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare LoveDA submission zip')
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                        help='Directory containing predicted PNG masks')
    parser.add_argument('-o', '--output-zip', type=str, default='submission.zip',
                        help='Output zip file path')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_zip = args.output_zip

    if not osp.exists(input_dir):
        print(f'[ERROR] Input directory not found: {input_dir}')
        return

    png_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if not png_files:
        print(f'[ERROR] No PNG files found in {input_dir}')
        return

    print(f'Found {len(png_files)} PNG files')

    tmp_dir = osp.join(input_dir, '_submit_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    count = 0
    for png_file in tqdm(png_files, desc='Processing'):
        img_path = osp.join(input_dir, png_file)

        # Strip Urban_ / Rural_ prefix
        new_name = png_file
        if new_name.startswith('Urban_'):
            new_name = new_name[len('Urban_'):]
        elif new_name.startswith('Rural_'):
            new_name = new_name[len('Rural_'):]

        # Read and convert to single-channel L mode
        img = Image.open(img_path)
        if img.mode == 'P':
            arr = np.array(img)
        elif img.mode == 'L':
            arr = np.array(img)
        else:
            arr = np.array(img.convert('L'))

        # Clip to valid class range 0-6
        arr = np.clip(arr, 0, 6).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')

        img.save(osp.join(tmp_dir, new_name))
        count += 1

    print(f'Processed {count} files')

    print(f'Creating zip: {output_zip}')
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in tqdm(sorted(os.listdir(tmp_dir)), desc='Zipping'):
            fpath = osp.join(tmp_dir, fname)
            zf.write(fpath, fname)

    import shutil
    shutil.rmtree(tmp_dir)

    print(f'\nDone! Submission zip: {output_zip}')
    print(f'  Total files: {count}')


if __name__ == '__main__':
    main()
