atc --singleop=./cast.json --output=./ --soc_version=Ascend310 --log info
atc --singleop=./transpose.json --output=./ --soc_version=Ascend310 --log info
atc --singleop=./argmax.json --output=./ --soc_version=Ascend310 --log info