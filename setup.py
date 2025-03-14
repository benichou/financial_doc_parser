from setuptools import setup, find_packages

# Read requirements.txt
with open('efs_parsing/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='financial_documents_parsing',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,  # Pass the requirements
    description='',
    author='Franck Benichou',
    author_email='franck.benichou@sciencespo.fr',
    url='https://github.com/benichou/financial_doc_parser',
)