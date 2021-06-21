#! /bin/bash

failed=0

for example in $(find $1 -maxdepth 1 -name "*.py")
do
    echo  -n "$x" $example '... '
    export DISPLAY= && python $example --fast_dev_run 1 &> run_log.txt
    if [[ $? -ne 0 ]]; then
        echo "FAILED!!"
        cat run_log.txt
        failed=1
    else
        echo "SUCCESS!"
    fi
    rm run_log.txt
done

exit $failed
