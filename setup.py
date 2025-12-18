"""
Setup configuration for CIBER analysis package.
"""

from setuptools import setup, find_packages
import os

# Read the README if it exists
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'CIBER Power Spectrum Analysis Package'

setup(
    name='ciber',
    version='1.0.0',
    author='Richard Feder',
    author_email='your.email@example.com',
    description='Power spectrum analysis pipeline for CIBER data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RichardFeder/ciber',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'matplotlib>=3.4',
        'astropy>=4.3',
        'pandas>=1.3',
        'pyfftw>=0.12',
        'scikit-learn>=0.24',
        'healpy>=1.15',
        'treecorr>=4.0',
        'reproject>=0.8',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.12',
            'sphinx>=4.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
        'torch': [
            'torch>=1.9',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
