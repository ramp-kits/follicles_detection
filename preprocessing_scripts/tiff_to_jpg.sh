# usage:
# ./preprocessing_scripts/tiff_to_jpg.sh
# (required imagemagick)
# Notes:
# https://www.imagemagick.org/script/architecture.php#tera-pixel
set -e
INPUT_FOLDER="data/coupes_tiff"
OUTPUT_FOLDER="data/coupes_jpg"
echo "Converting images from $INPUT_FOLDER to $OUTPUT_FOLDER"
mkdir -p $OUTPUT_FOLDER

for f in $INPUT_FOLDER/*.tiff
do  
    echo "Converting $f" 
    convert\
        -define registry:temporary-path=./data/tmp/ \
        -limit memory 500mb\
        $f\
        -set filename: "%t" $OUTPUT_FOLDER/%[filename:].jpg
done

echo "Done"
