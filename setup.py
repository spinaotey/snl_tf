from setuptools import setup

setup(
    # Needed to silence warnings
    name='SNL_TF',
    url='https://github.com/spinaotey/snl_tf',
    author='Sebastian Pina-Otey',
    author_email='spina@ifae.es',
    # Needed to actually package something
    packages=['snl_tf'],
    # Needed for dependencies
    #install_requires=['numpy','tensorflow','matplotlib'],
    # *strongly* suggested for sharing
    #version='0.5',
    license='MIT',
    description='Sequential Neural Likelihood using Tensorflow',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
