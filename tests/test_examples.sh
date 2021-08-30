#! /bin/bash


# Read Flags
gpu=0
while [ -n "$1" ]; do
	case "$1" in
	    --gpu) gpu=1;;
	    -g) gpu=1;;
        *) path=$1;;
	esac
	shift
done

python --version
echo "Using GPU: " $gpu

# Loop
failed=0

for example in $(find $path -maxdepth 1 -name "*.py")
do
    echo  -n "$x" $example '... '
    export DISPLAY= && python $example --fast_dev_run 1 --gpus $gpu &> run_log.txt
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
