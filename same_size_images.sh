for img in *.jpg; do
    convert -resize 750x “$img" "$img"
done
