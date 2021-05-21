#! /bin/bash

failed=0

for example in $(find $1 -maxdepth 1 -name "*.py")
do
    echo  -n "$x" $example '... '
    export DISPLAY= && python $example --fast_dev_run 1 &> /dev/null 
    if [[ $? -ne 0 ]]; then
        echo "Failed!!!!"
        failed=1
    else
        echo "Success."
    fi
done

exit $failed