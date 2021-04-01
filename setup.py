from setuptools import setup, find_packages

setup(
  name = 'jax_transformers',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Tranformers in pure JAX',
  author = 'Rodrigo Caridad',
  author_email = 'rorroart.code@gmail.com',
  url = 'https://github.com/RorroArt/jax-transformers',
  keywords = [
    'artificial intelligence',
    'natural language processing',
    'transformers'
  ],
  install_requires=[
    'numpy',
    'requests',
    'jax',
    'jaxlib',
    'tqdm'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
