from setuptools import setup, find_packages

setup(
    name='orientation',
    version='0.1',
    description='A simple example package',
    url='https://github.com/example/example',
    author='Your Name',
    author_email='your.name@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.15',
        'matplotlib>=3.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
