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

pip install pyspark


pip install --upgrade \
    jupyterlab==1.2.4 \
    ipywidgets \
    jupyterlab_latex \
    plotly \
    bokeh \
    numpy \
    scipy \
    numexpr \
    patsy \
    scikit-learn \
    scikit-image \
    matplotlib \
    ipython \
    pandas \
    sympy \
    seaborn \
    nose \
    jupyterlab-git && \
  jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/latex \
    @mflevine/jupyterlab_html \
    jupyterlab-drawio \
    @jupyterlab/plotly-extension \
    jupyterlab_bokeh \
    jupyterlab-spreadsheet \
    @jupyterlab/git


apt-get update

apt install --assume-yes openjdk-8-jdk

