from setuptools import setup, find_packages

def read_requirements(file):
    with open(file, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def read_file(file):
   with open(file, "r", encoding="utf-8") as f:
        return f.read()
    
long_description = read_file("README.md")
requirements = read_requirements("requirements.txt")

setup(
    name = 'imagedatasetanalyzer',

    version = '0.1.4',

    author = 'Joaquin Ortiz de Murua Ferrero',
    author_email = 'jortizdemuruaferrero@gmail.com',
    maintainer= 'Joaquin Ortiz de Murua Ferrero',
    maintainer_email= 'jortizdemuruaferrero@gmail.com',

    url='https://github.com/joortif/ImageDatasetAnalyzer',

    description = 'Image dataset analyzer using image embedding models and clustering methods.',

    long_description_content_type = 'text/markdown', 
    long_description = long_description,

    license = 'MIT license',

    packages = find_packages(exclude=["test"]), 
    install_requires = requirements,

    classifiers=[

        'Development Status :: 3 - Alpha',

        'Programming Language :: Python :: 3.10',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',       

        # Topics
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    keywords='instance semantic segmentation pytorch tensorflow huggingface opencv embedding image analysis machine learning deep learning active learning computer vision'
)