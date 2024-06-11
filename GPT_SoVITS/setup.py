import os 
from setuptools import setup, find_packages


cwd = os.path.dirname(os.path.abspath(__file__))

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='GPT_SoVITS',
    version='0.0.0',
    keywords=[
            'text-to-speech',
            'tts',
            'voice-clone',
            'zero-shot-tts'
    ],
    url='https://github.com/RVC-Boss/GPT-SoVITS',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=reqs,
    zip_safe=False,
)
