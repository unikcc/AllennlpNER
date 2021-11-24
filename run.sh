python=/home/libobo/.miniconda3/envs/allennlp24/bin/python
config_name="lstm"

if [ "$1" = "test" ];then
    $python -m allennlp evaluate checkpoint/$config_name/model.tar.gz data/conll2003/test.txt --include-package scripts
else
    rm -rf checkpoint/$config_name
    $python -m allennlp train training_config/$config_name.jsonnet --serialization-dir checkpoint/$config_name --include-package scripts
fi
