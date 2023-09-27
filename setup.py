from setuptools import setup, find_packages

setup(
    name='xia-fmri-workflow',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/dddd1007/xia_fmri_workflow',
    author='Xia Xiaokai',
    author_email='xia@xiaokai.me',
    description='My workflow of fMRI data analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'nipype',
        'toml',
        'matplotlib',
        'seaborn',
    ],
)
