#! /bin/bash

for example in $(find $1 -maxdepth 1 -name "*.py")
do
    echo "test:" $example
    export DISPLAY= && python $example --fast_dev_run 1
    if [[ $? -ne 0 ]]; then
        exit 1
    fi
done