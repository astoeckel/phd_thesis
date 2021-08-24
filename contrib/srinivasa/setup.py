from setuptools import setup, find_packages

setup(
    name='srinivasa',
    version='1.0.0',
    description='Pygments syntax highlighting style',
    author='Andreas Stöckel',
    packages=find_packages(),
    entry_points={
        'pygments.styles': [
            'srinivasa = srinivasa:SrinivasaStyle',
        ],
    },
)
