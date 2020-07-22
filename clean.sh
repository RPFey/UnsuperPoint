#!/bin/bash
cache=`find ./ -name __pycache__`
for file in $cache; do
    rm -r $file
done

cp -r ./output ../
rm -r ./output
rm -r ./.idea