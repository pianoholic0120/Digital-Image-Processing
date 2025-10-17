#!/bin/bash
python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_1_option1" \
    --output_file "../output/p1_1_option1_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_1_option2" \
    --output_file "../output/p1_1_option2_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_1_option3" \
    --output_file "../output/p1_1_option3_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_2_option1" \
    --output_file "../output/p1_2_option1_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_2_option2" \
    --output_file "../output/p1_2_option2_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_3_multi" \
    --output_file "../output/p1_3_multi_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_3_p6" \
    --output_file "../output/p1_3_p6_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_3_pinf" \
    --output_file "../output/p1_3_pinf_awb_evaluation.txt"

python3 evaluate_AWB.py \
    --test_images_dir "../images/test images" \
    --awb_results_dir "../output/p1_3_p8" \
    --output_file "../output/p1_3_p8_awb_evaluation.txt"