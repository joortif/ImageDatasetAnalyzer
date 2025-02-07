from setuptools import setup, find_packages
from codecs import open

# For installing PyTorch and Torchvision in Windows
import sys
import subprocess
import pkg_resources

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements("requirements.txt")

def remove_requirements(requirements, remove_elem):
    new_requirements = []
    for requirement in requirements:
        if remove_elem not in requirement:
            new_requirements.append(requirement)
    return new_requirements

def install_pytorch_for_gpu():
    """Try to install PyTorch and Torchvision with GPU support (CUDA)"""
    print('Checking for GPU support...')
    cuda_version = "cu116"  # Update this if you want to use a different CUDA version
    torch_version = f"torch==1.13.1+{cuda_version}"
    torchvision_version = f"torchvision==0.14.1+{cuda_version}"

    try:
        subprocess.check_call([
            'pip', 'install', torch_version, torchvision_version,
            '-f', 'https://download.pytorch.org/whl/torch_stable.html'
        ])
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch GPU version: {e}")
        print("Trying to install the CPU version instead...")
        install_pytorch_for_cpu()

def install_pytorch_for_cpu():
    """Fallback to install PyTorch and Torchvision with CPU support"""
    torch_version = "torch>=1.13.0,<2.0.0"
    torchvision_version = "torchvision>=0.14.0,<1.0.0"

    try:
        subprocess.check_call([
            'pip', 'install', torch_version, torchvision_version,
            '-f', 'https://download.pytorch.org/whl/torch_stable.html'
        ])
        print('Successfully installed PyTorch and Torchvision for CPU.')
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch CPU version: {e}")
        print("Please install PyTorch and Torchvision manually following the instructions at: https://pytorch.org/get-started/")

# Remove PyTorch and Torchvision from install requirements to handle manually
install_reqs = remove_requirements(install_reqs, 'torch')
install_reqs = remove_requirements(install_reqs, 'torchvision')

def check_and_install_pytorch():
    """Check if PyTorch and Torchvision are installed and compatible"""
    try:
        # Check if torch and torchvision are installed and compatible with the versions in requirements.txt
        pkg_resources.require("torch>=1.13.0,<2.0.0")
        pkg_resources.require("torchvision>=0.14.0,<1.0.0")
        print("Required versions of PyTorch and Torchvision are already installed.")
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
        print("PyTorch or Torchvision not found or incompatible, installing the correct versions...")
        install_pytorch_for_gpu()

print(f"Platform: {sys.platform}")
if sys.platform in ['win32', 'cygwin', 'windows']:
    # Check and install the correct versions of PyTorch and Torchvision if necessary
    check_and_install_pytorch() 

setup(
    name = 'imagedatasetanalyzer',

    version = '0.1.9',

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
    install_requires = install_reqs,
    include_package_data=True, 

    classifiers=[

        'Development Status :: 4 - Beta',

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