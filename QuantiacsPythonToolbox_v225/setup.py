from setuptools import setup

from quantiacsToolbox.version import __version__


setup(name='quantiacsToolbox',
      version=__version__,
      description='The Quantiacs Toolbox for trading system development',
      url='http://quantiacs.com/',
      author='Vernie Redmon',
      author_email='vnredmon@quantiacs.com',
      license='MIT',
      packages=['quantiacsToolbox','sampleSystems'],
      include_package_data = True,

      install_requires=[
        'pandas >=0.15.2',
        'numpy >= 1.9.2',
        'matplotlib >= 1.4.3',
      ],

      zip_safe=False,
     )
