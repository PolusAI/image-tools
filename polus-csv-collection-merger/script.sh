#!/bin/sh

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --input-collection-a)
    INPUT_A="$2"
    shift # past argument
    shift # past value
    ;;
    --input-collection-b)
    INPUT_B="$2"
    shift # past argument
    shift # past value
    ;;
    --append-a)
    APPEND_A="$2"
    shift # past argument
    shift # past value
    ;;
    --append-b)
    APPEND_B="$2"
    shift # past argument
    shift # past value
    ;;
    --output)
    OUTPUT="$2"
    shift # past argument
    shift # past value
    ;;
esac
done

echo "INPUT COLLECTION A  = ${INPUT_A}"
echo "INPUT COLLECTION B  = ${INPUT_B}"
echo "APPEND A  = ${APPEND_A}"
echo "APPEND B  = ${APPEND_B}"
echo "OUTPUT  = ${OUTPUT}"

COLLECTION_A="$(basename $INPUT_A)"
COLLECTION_B="$(basename $INPUT_B)"
echo "      "

echo "Copying files from collection A ($COLLECTION_A):"
for f in $INPUT_A/*; do echo "$(basename $f)"; done
if [ "$APPEND_A" = "true" ]; then
    for f in $INPUT_A/*; do cp "$f" "$OUTPUT"/"$COLLECTION_A"_"$(basename $f)"; done
else
    for f in $INPUT_A/*; do cp "$f" "$OUTPUT"/"$(basename $f)"; done
fi
echo "      "

echo "Copying files from collection B ($COLLECTION_B):"
for f in $INPUT_B/*; do echo "$(basename $f)"; done
if [ "$APPEND_B" = "true" ]; then
    for f in $INPUT_B/*; do cp "$f" "$OUTPUT"/"$COLLECTION_B"_"$(basename $f)"; done
else
    for f in $INPUT_B/*; do cp "$f" "$OUTPUT"/"$(basename $f)"; done
fi