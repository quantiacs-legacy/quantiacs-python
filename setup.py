from setuptools import setup

from quantiacsToolbox.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='quantiacsToolbox',
    version=__version__,
    author='Quantiacs',
    author_email='office@quantiacs.com',
    description='The Quantiacs Toolbox for trading system development',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://quantiacs.com/',
    packages=['quantiacsToolbox', 'sampleSystems'],
    download_url = 'https://github.com/Quantiacs/quantiacs-python/files/2288603/quantiacs-python-2.3.0.tar.gz',
    license='MIT',
    include_package_data = True,
    install_requires=['pandas >=0.15.2', 'numpy >= 1.14.0', 'matplotlib >= 2.1.0',],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
