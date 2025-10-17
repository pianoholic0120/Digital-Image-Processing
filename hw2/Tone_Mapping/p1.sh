#!/bin/bash
mkdir -p ../output/p1_1_option1_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_1_option1/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_1_option1_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_1_option1_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_1_option1_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_1_option2_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_1_option2/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_1_option2_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_1_option2_tone_mapping/${img}_metrics.txt"
        --output_curve "../output/p1_1_option2_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_1_option3_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_1_option3/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_1_option3_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_1_option3_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_1_option3_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_2_option1_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_2_option1/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_2_option1_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_2_option1_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_2_option1_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_2_option2_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_2_option2/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_2_option2_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_2_option2_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_2_option2_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_3_multi_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_3_multi/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_3_multi_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_3_multi_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_3_multi_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_3_p6_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_3_p6/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_3_p6_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_3_p6_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_3_p6_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_3_pinf_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_3_pinf/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_3_pinf_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_3_pinf_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_3_pinf_tone_mapping/${img}_curve.png"
    echo ""
done

mkdir -p ../output/p1_3_p8_tone_mapping/

for img in a b c d e; do
    echo "Processing image: ${img}"
    python3 p1.py \
        --source_image "../output/p1_3_p8/${img}.png" \
        --reference_image "../images/reference images/${img}_reference.tiff" \
        --output_image "../output/p1_3_p8_tone_mapping/${img}_tone_mapped.png" \
        --output_metrics "../output/p1_3_p8_tone_mapping/${img}_metrics.txt" \
        --output_curve "../output/p1_3_p8_tone_mapping/${img}_curve.png"
    echo ""
done

echo "All images processed!"

