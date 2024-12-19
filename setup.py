from setuptools import setup, find_packages

setup(
    name='youtube-video-pipeline',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'google-generativeai',
        'watchdog',
        'opencv-python',
        'ffmpeg-python',
        'numpy',
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'vidpipe=src.cli:cli',
        ],
    },
    author='Giovanni',
    description='Automated YouTube video processing pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
