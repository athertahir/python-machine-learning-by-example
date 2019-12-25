# pip install --upgrade pip

# pip install keras
# pip install Pillow

echo 'Installing Dependencies'

pip install numpy==1.16.0
pip install nltk spacy
pip install --upgrade gensim
pip install -U textblob
pip install gym


pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

python -m nltk.downloader all

# conda install -c conda-forge nltk_data -y

sudo pip install pyspark


sudo pip install --upgrade jupyterlab-git

sudo jupyter labextension install @jupyterlab/git


sudo jupyter serverextension enable --py jupyterlab_git

apt-get update

apt install --assume-yes openjdk-8-jdk

