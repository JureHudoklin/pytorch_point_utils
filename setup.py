from setuptools import find_packages, setup

setup(
    name='pytorch_point_utils',
    version='0.1.0',
    author='Jure Hudoklin',
    author_email='jure.hudoklin97@gmail.com',
    description='A library for working with point clouds in PyTorch',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)