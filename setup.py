# setup.py
from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='xavierconcepts',
      description="Reproduction of Paper: Grounding Counterfactual Explanation of Image Classifiers to Textual Concept Space",
      packages=find_packages(), # NEW: find packages automatically
      install_requires=requirements) # NEW