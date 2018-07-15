FROM josephkphan/tensorflow-base:latest

WORKDIR "./"

COPY    download.py imdb.py sentiment-analysis.py ./

# CMD [ "jupyter notebook", "--allow-root" ]
CMD     [ "python3 ", "sentiment-analysis.py" ]