#!/bin/bash

for i in `ls`
do
    if [[ $i == *.metircs ]];
    then
        echo $i
        echo ${i: :-8}
        mv $i ${i: :-8}".metrics"
    fi
done
