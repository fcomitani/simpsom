from setuptools import setup, find_packages
from codecs import open
from os import path

here    = path.abspath(path.dirname(__file__))
version = open("simpsom/_version.py").readlines()[-1].split()[-1].strip("\"'")

with open(path.join(here, 'long.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='simpsom',

    version=version,

    description='Simple Self-Organizing Maps in Python',
    long_description=long_description,

    url='https://github.com/fcomitani/simpsom',
	download_url = 'https://github.com/fcomitani/simpsom/archive/'+version+'.tar.gz', 
    author='Federico Comitani',
    author_email='federico.comitani@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
	'Programming Language :: Python :: 3'
		],

    keywords='kohonen self-organizing maps, self-organizing maps, clustering ,dimension-reduction, som',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['numpy>=1.19.5',
		'scikit-learn>=0.22.2.post1',
		'matplotlib>=3.3.3'],

    extras_require={ 'gpu': ['cupy==8.60',
				'cuml==0.18']},
    zip_safe=False,

)

