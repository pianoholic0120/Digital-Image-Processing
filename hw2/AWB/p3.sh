#!/bin/bash
# python3 p3.py  --input_image "../images/test images/a.tif" --illuminant_file "../images/test images/a.rgb" --output_dir ../output/p1_3/
# python3 p3.py  --input_image "../images/test images/b.tif" --illuminant_file "../images/test images/b.rgb" --output_dir ../output/p1_3/
# python3 p3.py  --input_image "../images/test images/c.tif" --illuminant_file "../images/test images/c.rgb" --output_dir ../output/p1_3/
# python3 p3.py  --input_image "../images/test images/d.tif" --illuminant_file "../images/test images/d.rgb" --output_dir ../output/p1_3/
# python3 p3.py  --input_image "../images/test images/e.tif" --illuminant_file "../images/test images/e.rgb" --output_dir ../output/p1_3/

# Default from paper p = 6
python3 p3.py  --input_image "../images/test images/a.tif" --output_dir ../output/p1_3_p6/
python3 p3.py  --input_image "../images/test images/b.tif" --output_dir ../output/p1_3_p6/
python3 p3.py  --input_image "../images/test images/c.tif" --output_dir ../output/p1_3_p6/
python3 p3.py  --input_image "../images/test images/d.tif" --output_dir ../output/p1_3_p6/
python3 p3.py  --input_image "../images/test images/e.tif" --output_dir ../output/p1_3_p6/

# Max RGB p = ∞
python3 p3.py  --input_image "../images/test images/a.tif" --output_dir ../output/p1_3_pinf/ --method max_rgb
python3 p3.py  --input_image "../images/test images/b.tif" --output_dir ../output/p1_3_pinf/ --method max_rgb
python3 p3.py  --input_image "../images/test images/c.tif" --output_dir ../output/p1_3_pinf/ --method max_rgb
python3 p3.py  --input_image "../images/test images/d.tif" --output_dir ../output/p1_3_pinf/ --method max_rgb
python3 p3.py  --input_image "../images/test images/e.tif" --output_dir ../output/p1_3_pinf/ --method max_rgb

# multi-scale approach
python3 p3.py  --input_image "../images/test images/a.tif" --output_dir ../output/p1_3_multi/ --method multi_scale
python3 p3.py  --input_image "../images/test images/b.tif" --output_dir ../output/p1_3_multi/ --method multi_scale
python3 p3.py  --input_image "../images/test images/c.tif" --output_dir ../output/p1_3_multi/ --method multi_scale
python3 p3.py  --input_image "../images/test images/d.tif" --output_dir ../output/p1_3_multi/ --method multi_scale
python3 p3.py  --input_image "../images/test images/e.tif" --output_dir ../output/p1_3_multi/ --method multi_scale

# Use different p values
python3 p3.py  --input_image "../images/test images/a.tif" --output_dir ../output/p1_3_p8/ --p 8
python3 p3.py  --input_image "../images/test images/b.tif" --output_dir ../output/p1_3_p8/ --p 8
python3 p3.py  --input_image "../images/test images/c.tif" --output_dir ../output/p1_3_p8/ --p 8
python3 p3.py  --input_image "../images/test images/d.tif" --output_dir ../output/p1_3_p8/ --p 8
python3 p3.py  --input_image "../images/test images/e.tif" --output_dir ../output/p1_3_p8/ --p 8