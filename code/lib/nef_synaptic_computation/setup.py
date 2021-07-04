from setuptools import setup, find_packages

setup(
    name='nef_synaptic_computation',
    packages=[
        'nef_synaptic_computation'
    ],
    package_data={
        'nef_synaptic_computation':
        ['Makefile', 'two_compartment_lif.cpp']
    },
    version='0.1',
    author='Andreas Stöckel',
    description='Synaptic computation in the NEF',
    url='https://github.com/ctn-waterloo/nef_synaptic_computation',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ])

