# This runs an interactive shell of the sentiment-analysis container
docker run -v /Users/josephphan/Desktop/SchoolWork/TensorFlowResearch/tensor-python/sentiment-analysis:/notebooks -it --entrypoint /bin/bash josephkphan/tensorflow-sentiment-analysis:latest