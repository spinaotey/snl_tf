# Sequential Neural Likelihood estimation - Tensorflow implementation

This repository's aim is to adapt G. Papamakarios' [Sequantial Neural Likelihood](https://github.com/gpapamak/snl/) method from Theano to Tensorflow.

The method heavily depends on the Masked Autoregressive Flow, which has been already translated into Tensorflow in my other [repository](https://github.com/spinaotey/maf_tf).

#### How to setup the snl_tf Python library.

This is a Python 3 library (current version used 3.6+). In order to install this library first install the required packages listed in requirements.txt (```pip install -r requirements.txt)```). Then just use ```pip install -e .``` in the main folder of the repository (where the setup.py is situated). After that, you should be able to import the package as usual with ```import snl_tf```.