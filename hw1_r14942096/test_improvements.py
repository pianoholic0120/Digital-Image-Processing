import subprocess
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def run_demosaicing_tests():
    """Run all demosaicing methods and generate results"""
    # Test images
    test_images = ["A.tiff", "B.png", "C.png", "D.png", "E.png"]
    input_dir = "./images/raw_image/"
    
    # Methods to test
    methods = [
        ("p1", "python3 p1.py", "./output/p1/"),
        ("p2_3", "python3 p2.py --threshold 3", "./output/p2_3thre/"),
        ("p2_5", "python3 p2.py --threshold 5", "./output/p2_5thre/"), 
        ("p2_10", "python3 p2.py --threshold 10", "./output/p2_10thre/"),
        ("p2_20", "python3 p2.py --threshold 20", "./output/p2_20thre/"),
        ("p2_30", "python3 p2.py --threshold 30", "./output/p2_30thre/"),
        ("p2_50", "python3 p2.py --threshold 50", "./output/p2_50thre/"),
        ("p3", "python3 p3.py", "./output/p3/")
    ]
    
    print("Running demosaicing tests...")
    print("=" * 60)
    
    for method_name, command, output_dir in methods:
        print(f"\nTesting {method_name}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for image in test_images:
            input_path = os.path.join(input_dir, image)
            if os.path.exists(input_path):
                try:
                    # Run the demosaicing command
                    cmd = f"{command} --input_image {input_path}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"  ✓ {image} processed successfully")
                    else:
                        print(f"  ✗ {image} failed: {result.stderr}")
                        
                except Exception as e:
                    print(f"  ✗ {image} error: {str(e)}")
            else:
                print(f"  ⚠ {image} not found")
    
    print("\n" + "=" * 60)
    print("All demosaicing tests completed!")
    return methods

def calculate_all_psnr():
    """Calculate PSNR for all methods and images"""
    # Test images
    test_images = ["A", "B", "C", "D", "E"]
    gt_dir = "./images/ground_truth/"
    
    # Methods and their output directories
    methods = [
        ("p1", "./output/p1/"),
        ("p2_3", "./output/p2_3thre/"),
        ("p2_5", "./output/p2_5thre/"), 
        ("p2_10", "./output/p2_10thre/"),
        ("p2_20", "./output/p2_20thre/"),
        ("p2_30", "./output/p2_30thre/"),
        ("p2_50", "./output/p2_50thre/"),
        ("p3", "./output/p3/")
    ]
    
    print("\nCalculating PSNR values...")
    print("=" * 60)
    
    # Store results
    results = []
    
    for method_name, output_dir in methods:
        print(f"\nProcessing {method_name}...")
        method_results = []
        
        for image in test_images:
            gt_path = os.path.join(gt_dir, f"{image}.png")
            recon_path = os.path.join(output_dir, f"{image}.png")
            
            if os.path.exists(gt_path) and os.path.exists(recon_path):
                try:
                    # Load images
                    gt = cv2.imread(gt_path)
                    recon = cv2.imread(recon_path)
                    
                    if gt is not None and recon is not None:
                        # Calculate PSNR
                        psnr_value = calculate_psnr(recon, gt)
                        method_results.append(psnr_value)
                        results.append({
                            'Method': method_name,
                            'Image': image,
                            'PSNR': psnr_value
                        })
                        print(f"  {image}: {psnr_value:.2f} dB")
                    else:
                        print(f"  {image}: Error loading images")
                        method_results.append(np.nan)
                except Exception as e:
                    print(f"  {image}: Error - {str(e)}")
                    method_results.append(np.nan)
            else:
                print(f"  {image}: Missing files")
                method_results.append(np.nan)
        
        # Calculate average PSNR for this method
        valid_psnr = [p for p in method_results if not np.isnan(p)]
        if valid_psnr:
            avg_psnr = np.mean(valid_psnr)
            print(f"  Average PSNR: {avg_psnr:.2f} dB")
        else:
            print(f"  Average PSNR: N/A")
    
    return results

def create_visualizations(results_df):
    """Create comprehensive visualizations"""
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create output directory for plots
    plot_dir = "./output/analysis_plots/"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. PSNR Comparison Bar Chart
    plt.figure(figsize=(15, 8))
    pivot_df = results_df.pivot(index='Image', columns='Method', values='PSNR')
    pivot_df.plot(kind='bar', figsize=(15, 8))
    plt.title('PSNR Comparison Across Methods and Images', fontsize=16, fontweight='bold')
    plt.xlabel('Image', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'psnr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Method Performance Heatmap
    plt.figure(figsize=(12, 8))
    pivot_df_clean = pivot_df.dropna()
    
    # Create heatmap using matplotlib
    im = plt.imshow(pivot_df_clean.values, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='PSNR (dB)')
    
    # Set ticks and labels
    plt.xticks(range(len(pivot_df_clean.columns)), pivot_df_clean.columns, rotation=45)
    plt.yticks(range(len(pivot_df_clean.index)), pivot_df_clean.index)
    
    # Add text annotations
    for i in range(len(pivot_df_clean.index)):
        for j in range(len(pivot_df_clean.columns)):
            text = plt.text(j, i, f'{pivot_df_clean.iloc[i, j]:.1f}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    plt.title('PSNR Heatmap: Methods vs Images', fontsize=16, fontweight='bold')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Image', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'psnr_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Method Average Performance
    plt.figure(figsize=(12, 6))
    method_avg = results_df.groupby('Method')['PSNR'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(method_avg)), method_avg.values, color='skyblue', edgecolor='navy')
    plt.title('Average PSNR by Method', fontsize=16, fontweight='bold')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Average PSNR (dB)', fontsize=12)
    plt.xticks(range(len(method_avg)), method_avg.index, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'method_average.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Box Plot for Method Distribution
    plt.figure(figsize=(12, 8))
    
    # Create box plot manually
    methods = results_df['Method'].unique()
    data_by_method = [results_df[results_df['Method'] == method]['PSNR'].values for method in methods]
    
    bp = plt.boxplot(data_by_method, labels=methods, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('PSNR Distribution by Method', fontsize=16, fontweight='bold')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'psnr_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Image-wise Performance
    plt.figure(figsize=(12, 8))
    image_avg = results_df.groupby('Image')['PSNR'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(image_avg)), image_avg.values, color='lightcoral', edgecolor='darkred')
    plt.title('Average PSNR by Image', fontsize=16, fontweight='bold')
    plt.xlabel('Image', fontsize=12)
    plt.ylabel('Average PSNR (dB)', fontsize=12)
    plt.xticks(range(len(image_avg)), image_avg.index)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'image_average.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizations saved in: {plot_dir}")

def quantitative_analysis(results_df):
    """Perform quantitative analysis"""
    print("\n" + "=" * 60)
    print("QUANTITATIVE ANALYSIS")
    print("=" * 60)
    
    # 1. Overall Statistics
    print("\n1. OVERALL STATISTICS")
    print("-" * 30)
    print(f"Total measurements: {len(results_df)}")
    print(f"Mean PSNR: {results_df['PSNR'].mean():.2f} dB")
    print(f"Std PSNR: {results_df['PSNR'].std():.2f} dB")
    print(f"Min PSNR: {results_df['PSNR'].min():.2f} dB")
    print(f"Max PSNR: {results_df['PSNR'].max():.2f} dB")
    
    # 2. Method Rankings
    print("\n2. METHOD RANKINGS (by average PSNR)")
    print("-" * 40)
    method_stats = results_df.groupby('Method')['PSNR'].agg(['mean', 'std', 'min', 'max']).round(2)
    method_stats = method_stats.sort_values('mean', ascending=False)
    
    for i, (method, stats) in enumerate(method_stats.iterrows(), 1):
        print(f"{i:2d}. {method:8s}: {stats['mean']:6.2f} ± {stats['std']:5.2f} dB "
              f"(range: {stats['min']:.2f} - {stats['max']:.2f})")
    
    # 3. Best and Worst Performances
    print("\n3. BEST AND WORST PERFORMANCES")
    print("-" * 35)
    best_idx = results_df['PSNR'].idxmax()
    worst_idx = results_df['PSNR'].idxmin()
    
    best_row = results_df.loc[best_idx]
    worst_row = results_df.loc[worst_idx]
    
    print(f"Best:  {best_row['Method']:8s} on {best_row['Image']} = {best_row['PSNR']:.2f} dB")
    print(f"Worst: {worst_row['Method']:8s} on {worst_row['Image']} = {worst_row['PSNR']:.2f} dB")
    
    # 4. Method Comparison
    print("\n4. METHOD COMPARISON")
    print("-" * 25)
    p3_psnr = results_df[results_df['Method'] == 'p3']['PSNR'].mean()
    p1_psnr = results_df[results_df['Method'] == 'p1']['PSNR'].mean()
    p2_best = results_df[results_df['Method'].str.startswith('p2')]['PSNR'].max()
    
    print(f"P3 (Stochastic):     {p3_psnr:.2f} dB")
    print(f"P1 (Simple):         {p1_psnr:.2f} dB")
    print(f"P2 (Edge-aware):     {p2_best:.2f} dB")
    print(f"P3 vs P1 improvement: {p3_psnr - p1_psnr:+.2f} dB")
    print(f"P3 vs P2 improvement: {p3_psnr - p2_best:+.2f} dB")
    
    # 5. Image Difficulty Analysis
    print("\n5. IMAGE DIFFICULTY ANALYSIS")
    print("-" * 30)
    image_stats = results_df.groupby('Image')['PSNR'].agg(['mean', 'std']).round(2)
    image_stats = image_stats.sort_values('mean', ascending=False)
    
    for image, stats in image_stats.iterrows():
        print(f"{image}: {stats['mean']:6.2f} ± {stats['std']:5.2f} dB")
    
    # 6. Statistical Significance (if we have enough data)
    print("\n6. STATISTICAL INSIGHTS")
    print("-" * 25)
    p3_data = results_df[results_df['Method'] == 'p3']['PSNR']
    p1_data = results_df[results_df['Method'] == 'p1']['PSNR']
    
    if len(p3_data) > 1 and len(p1_data) > 1:
        improvement = p3_data.mean() - p1_data.mean()
        print(f"P3 shows {improvement:+.2f} dB average improvement over P1")
        
        if improvement > 0:
            print("✓ P3 (Stochastic) method is superior to P1 (Simple)")
        else:
            print("⚠ P3 method needs improvement")
    
    # Save detailed results
    results_df.to_csv('./output/analysis_plots/detailed_results.csv', index=False)
    method_stats.to_csv('./output/analysis_plots/method_statistics.csv')
    image_stats.to_csv('./output/analysis_plots/image_statistics.csv')
    
    print(f"\nDetailed results saved in: ./output/analysis_plots/")

def run_demosaicing_test():
    """Main function to run complete analysis"""
    print("COMPREHENSIVE DEMOSAICING ANALYSIS")
    print("=" * 60)
    
    # Step 1: Run all demosaicing methods
    print("Step 1: Running all demosaicing methods...")
    methods = run_demosaicing_tests()
    
    # Step 2: Calculate PSNR values
    print("\nStep 2: Calculating PSNR values...")
    results = calculate_all_psnr()
    
    if not results:
        print("No results found. Please check if all methods completed successfully.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Step 3: Create visualizations
    print("\nStep 3: Creating visualizations...")
    create_visualizations(results_df)
    
    # Step 4: Perform quantitative analysis
    print("\nStep 4: Performing quantitative analysis...")
    quantitative_analysis(results_df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check the following outputs:")
    print("- ./output/analysis_plots/ - All visualization charts")
    print("- ./output/analysis_plots/detailed_results.csv - Raw PSNR data")
    print("- ./output/analysis_plots/method_statistics.csv - Method statistics")
    print("- ./output/analysis_plots/image_statistics.csv - Image statistics")

if __name__ == "__main__":
    run_demosaicing_test()

