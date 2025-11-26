import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import os

def create_chart_image(data, title, width=1200, height=800):
    """Create a simple chart using PIL"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw title
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (width - (bbox[2] - bbox[0])) // 2
    draw.text((title_x, 20), title, fill='black', font=title_font)
    
    # Chart area
    margin = 80
    chart_x = margin
    chart_y = 100
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin
    
    # Draw axes
    draw.line([(chart_x, chart_y), (chart_x, chart_y + chart_h)], fill='black', width=2)
    draw.line([(chart_x, chart_y + chart_h), (chart_x + chart_w, chart_y + chart_h)], fill='black', width=2)
    
    return img, draw, chart_x, chart_y, chart_w, chart_h, label_font, small_font

# Load statistics
with open('results/comprehensive_experiments/statistics.json', 'r') as f:
    stats = json.load(f)

kernel_sizes = stats['kernel_sizes']
original_white = 443646

# Create comparison table image
width, height = 1400, 1000
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

try:
    title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    cell_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except:
    title_font = ImageFont.load_default()
    header_font = ImageFont.load_default()
    cell_font = ImageFont.load_default()

# Title
title = "Morphological Operations - Quantitative Analysis"
bbox = draw.textbbox((0, 0), title, font=title_font)
draw.text(((width - (bbox[2] - bbox[0])) // 2, 30), title, fill='black', font=title_font)

# Table headers
y_start = 120
x_positions = [50, 200, 400, 600, 800, 1000, 1200]
headers = ["Kernel", "Erosion", "Dilation", "Opening", "Closing", "Op/Ef", "Cl/Di"]
for i, header in enumerate(headers):
    draw.text((x_positions[i], y_start), header, fill='black', font=header_font)

# Table data
y = y_start + 50
for i, k_size in enumerate(kernel_sizes):
    erosion_removed = stats['erosion'][i]['removed']
    dilation_added = stats['dilation'][i]['added']
    opening_removed = stats['opening'][i]['removed']
    closing_added = stats['closing'][i]['added']
    
    op_eff = (opening_removed / erosion_removed * 100) if erosion_removed > 0 else 0
    cl_eff = (closing_added / dilation_added * 100) if dilation_added > 0 else 0
    
    row_data = [
        f"{k_size}×{k_size}",
        f"{erosion_removed:,} ({stats['erosion'][i]['removal_rate']:.2f}%)",
        f"{dilation_added:,} ({stats['dilation'][i]['addition_rate']:.2f}%)",
        f"{opening_removed:,} ({stats['opening'][i]['removal_rate']:.2f}%)",
        f"{closing_added:,} ({stats['closing'][i]['addition_rate']:.2f}%)",
        f"{op_eff:.1f}%",
        f"{cl_eff:.1f}%"
    ]
    
    for j, data in enumerate(row_data):
        draw.text((x_positions[j], y), data, fill='black', font=cell_font)
    
    y += 35

# Save
img.save('results/comprehensive_experiments/figures/quantitative_analysis.png')
print("Quantitative analysis chart created!")

