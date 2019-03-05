"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='tensorloaders',  # Required
    version='0.0.1',  # Required
    description='In-memory tensorloaders for deep learning datasets',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/deeptechlabsco/tensorloaders',  # Optional
    author='Dendi Suhubdy',  # Optional
    author_email='dendi@deeptechlabs.co',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD-3 Clause License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='deeplearning scientific mathematics',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=['numpy', 'redis'],  # Optional
    extras_require={  # Optional
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    entry_points={  # Optional
        'console_scripts': [
            'tensorloaders=tensorloaders:main',
        ],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/deeptechlabs/tensorloaders/issues',
        'Funding': 'http://deeptechlabs.co',
        'Say Thanks!': 'http://deeptechlabs.co/research',
        'Source': 'https://github.com/deeptechlabs/tensorloaders',
    },
)
