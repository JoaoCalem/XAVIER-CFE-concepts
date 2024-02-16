# setup.py
from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='xavierconcepts',
    description="Reproduction of Paper: Grounding Counterfactual Explanation of Image Classifiers to Textual Concept Space",
    packages=find_packages(),
    package_data={'xavierconcepts': ['concepts/']},
    include_package_data=True,
    install_requires=requirements)