# pip install --upgrade pip

# pip install keras
# pip install Pillow

echo 'Installing Dependencies'

sudo pip install numpy==1.16.0
sudo pip install nltk spacy
sudo pip install --upgrade gensim
sudo pip install -U textblob
sudo pip install matplotlib
sudo pip install gym
sudo pip install matplotlib_venn 
sudo pip install tensorflow

sudo pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

sudo python -m nltk.downloader all

# conda install -c conda-forge nltk_data -y

sudo pip install pyspark

sudo apt-get update

sudo apt install --assume-yes openjdk-8-jdk