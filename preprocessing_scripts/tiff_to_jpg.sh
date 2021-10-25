INPUT_FOLDER="data/coupes_tiff"
OUTPUT_FOLDER="data/coupes_jpg"
echo "Converting images from $INPUT_FOLDER to $OUTPUT_FOLDER"
mkdir -p $OUTPUT_FOLDER

for f in $INPUT_FOLDER/*.tiff
do  
    echo "Converting $f" 
    convert $f -set filename: "%t" $OUTPUT_FOLDER/%[filename:].jpg
done

echo "Done"
