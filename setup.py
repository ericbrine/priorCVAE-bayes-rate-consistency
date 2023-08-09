try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


packages = [
    'bayes_rate_consistency'
 ]

setup(name='bayes_rate_consistency',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Prior CVAE Bayes Rate Consistency',
      author='MLGH',
      packages=packages
      )
