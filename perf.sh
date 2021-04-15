#!/bin/zsh

dir=~/datasets/unsplash

i=1
imsizes="256,512,724,1024,1448,2048,2560,3072"
iters="500,400,300,200,150,100,75,50"

for size in 256 512 724 1024 1200 1448 1600 1800 2048; do

echo $size

echo optex-pca
/usr/bin/time python -u ~/code/OptimalTextures/optex.py --output_dir $dir/results/optex --size $size \
    -s $dir/pawel-czerwinski-Mxi0EdBIpZk-unsplash.jpg --hist_mode pca

echo optex-cdf
/usr/bin/time python -u ~/code/OptimalTextures/optex.py --output_dir $dir/results/optex --size $size \
    -s $dir/pawel-czerwinski-Mxi0EdBIpZk-unsplash.jpg --hist_mode cdf

echo maua
/usr/bin/time python -u ~/code/maua-style/style.py \
    -ffmpeg_args ~/code/maua-style/config/ffmpeg-hevc.json \
    -load_args ~/code/maua-style/config/args-img.json \
    -scaling_args ~/code/maua-style/config/scaling-img.json \
    -output_dir $dir/results/maua -gpu 0  -uniq\
    -image_sizes $(echo $imsizes | cut -d, -f-$i) -num_iters $(echo $iters | cut -d, -f-$i) \
    -init random -content random -style $dir/pawel-czerwinski-Mxi0EdBIpZk-unsplash.jpg

echo embark
/usr/bin/time texture-synthesis --in-size $size --out-size $size \
    --out $dir/results/embark/pawel-czerwinski-Mxi0EdBIpZk-${size}.png \
    generate $dir/pawel-czerwinski-Mxi0EdBIpZk-unsplash.jpg

i=$((i+1))

done

exit

i=1
for size in 256 512 724 1024 1448 2048 2560 3072; do

echo $size

echo optex
/usr/bin/time python -u ~/code/OptimalTextures/optex.py --output_dir $dir/results/optex --size $size \
    -c $dir/nathan-dumlao-IWnkyagQPXw-unsplash.jpg -s $dir/philipp-potocnik-5rsNohd8bY8-unsplash.jpg

echo maua
/usr/bin/time python -u ~/code/maua-style/style.py \
    -ffmpeg_args ~/code/maua-style/config/ffmpeg-hevc.json \
    -load_args ~/code/maua-style/config/args-img.json \
    -scaling_args ~/code/maua-style/config/scaling-img.json \
    -output_dir $dir/results/maua -gpu 0  -uniq\
    -image_sizes $(echo $imsizes | cut -d, -f-$i) -num_iters $(echo $iters | cut -d, -f-$i) \
    -init random -content $dir/nathan-dumlao-IWnkyagQPXw-unsplash.jpg -style $dir/philipp-potocnik-5rsNohd8bY8-unsplash.jpg

echo embark
/usr/bin/time texture-synthesis --in-size $(echo "2 / 3 * $size" | bc -l | awk '{print int($1+0.5)}' )x$size --out-size $(echo "2 / 3 * $size" | bc -l| awk '{print int($1+0.5)}' )x$size --out $dir/results/embark/nathan-dumlao-IWnkyagQPXw-philipp-potocnik-5rsNohd8bY8-${size}.png \
    transfer-style --guide $dir/nathan-dumlao-IWnkyagQPXw-unsplash.jpg --style $dir/philipp-potocnik-5rsNohd8bY8-unsplash.jpg

i=$((i+1))

done

i=1
for size in 256 512 724 1024 1448 2048 2560 3072; do

echo $size

echo optex
/usr/bin/time python -u ~/code/OptimalTextures/optex.py --output_dir $dir/results/optex --size $size \
    -s $dir/ciaran-o-brien-ITu-L0FuPPk-unsplash-small.jpg $dir/pawel-nolbert-4u2U8EO9OzY-unsplash.jpg

echo maua
/usr/bin/time python -u ~/code/maua-style/style.py \
    -ffmpeg_args ~/code/maua-style/config/ffmpeg-hevc.json \
    -load_args ~/code/maua-style/config/args-img.json \
    -scaling_args ~/code/maua-style/config/scaling-img.json \
    -output_dir $dir/results/maua -gpu 0  -uniq\
    -image_sizes $(echo $imsizes | cut -d, -f-$i) -num_iters $(echo $iters | cut -d, -f-$i) \
    -init random -content random -style $dir/ciaran-o-brien-ITu-L0FuPPk-unsplash-small.jpg $dir/pawel-nolbert-4u2U8EO9OzY-unsplash.jpg

echo embark
/usr/bin/time texture-synthesis --in-size $size --out-size $size \
    --out $dir/results/embark/ciaran-o-brien-ITu-L0FuPPk-unsplash-small-pawel-nolbert-4u2U8EO9OzY-unsplash-${size}.png \
    generate $dir/ciaran-o-brien-ITu-L0FuPPk-unsplash-small.jpg $dir/pawel-nolbert-4u2U8EO9OzY-unsplash.jpg

i=$((i+1))

done

size=1024
python -u ~/code/OptimalTextures/optex.py --output_dir $dir/results/optex --size $size \
    -s $dir/franz-schekolin-IOiW0iGKwQg-unsplash.jpg -c $dir/v-srinivasan-h64wUnq6ZxM-unsplash.jpg

python -u ~/code/OptimalTextures/optex.py --output_dir $dir/results/optex --size $size \
    -c $dir/franz-schekolin-IOiW0iGKwQg-unsplash.jpg -s $dir/v-srinivasan-h64wUnq6ZxM-unsplash.jpg

python -u ~/code/maua-style/style.py \
    -ffmpeg_args ~/code/maua-style/config/ffmpeg-hevc.json \
    -load_args ~/code/maua-style/config/args-img.json \
    -scaling_args ~/code/maua-style/config/scaling-img.json \
    -output_dir $dir/results/maua \
    -image_sizes 256,512,724,1024 -num_iters 500,400,300,200 \
    -init random -content $dir/franz-schekolin-IOiW0iGKwQg-unsplash.jpg -style $dir/v-srinivasan-h64wUnq6ZxM-unsplash.jpg

python -u ~/code/maua-style/style.py \
    -ffmpeg_args ~/code/maua-style/config/ffmpeg-hevc.json \
    -load_args ~/code/maua-style/config/args-img.json \
    -scaling_args ~/code/maua-style/config/scaling-img.json \
    -output_dir $dir/results/maua \
    -image_sizes 256,512,724,1024 -num_iters 500,400,300,200 \
    -init random -style $dir/franz-schekolin-IOiW0iGKwQg-unsplash.jpg -content $dir/v-srinivasan-h64wUnq6ZxM-unsplash.jpg

texture-synthesis --in-size $(echo "1.555 * $size" | bc -l | awk '{print int($1+0.5)}' )x$size --out-size $(echo "1.555 * $size" | bc -l | awk '{print int($1+0.5)}' )x$size --out $dir/results/embark/franz-schekolin-IOiW0iGKwQg-v-srinivasan-h64wUnq6ZxM.png \
    transfer-style --guide $dir/franz-schekolin-IOiW0iGKwQg-unsplash.jpg --style $dir/v-srinivasan-h64wUnq6ZxM-unsplash.jpg

texture-synthesis --in-size $(echo "1.555 * $size" | bc -l | awk '{print int($1+0.5)}' )x$size --out-size $(echo "1.555 * $size" | bc -l | awk '{print int($1+0.5)}' )x$size --out $dir/results/embark/v-srinivasan-h64wUnq6ZxM-franz-schekolin-IOiW0iGKwQg.png \
    transfer-style --guide $dir/v-srinivasan-h64wUnq6ZxM-unsplash.jpg --style $dir/franz-schekolin-IOiW0iGKwQg-unsplash.jpg
