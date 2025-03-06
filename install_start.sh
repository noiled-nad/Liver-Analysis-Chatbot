mkdir ./model

cd ./model

export HF_ENDPOINT=https://hf-mirror.com

wget https://hf-mirror.com/hfd/hfd.sh

chmod a+x hfd.sh


./hfd.sh ckiplab/bert-base-chinese-ner

./hfd.sh deepseek-ai/Janus-Pro-7B




