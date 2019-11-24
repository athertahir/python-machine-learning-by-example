# pip install --upgrade pip
# pip install matplotlib_venn 


# pip install keras
# pip install Pillow

echo 'Installing Dependencies'

pip install numpy==1.16.0
pip install nltk spacy

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

python -m nltk.downloader all

# conda install -c conda-forge nltk_data -y
