#! /bin/bash

failed=0

for example in $(find $1 -maxdepth 1 -name "*.py")
do
    echo  -n "$x" $example '... '
    export DISPLAY= && python $example --fast_dev_run 1 &> /dev/null
    if [[ $? -ne 0 ]]; then
        echo "FAILED!!"
        failed=1
    else
        echo "SUCCESS!"
    fi
done

exit $failed
