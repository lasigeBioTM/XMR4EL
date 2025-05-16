from setuptools import setup, find_packages


def parse_requirements(filename):
    """Read the requirements file and return a list of dependencies"""
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines() if line.strip() and not line.startswith('#')]

setup(
    name='xmr4el',
    version='0.1',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    description='eXtreme Multi-Label Ranking for Entity Linking',
    author='João Vedor',
    author_email='fc56311@alunos.fc.ul.pt',
    url='https://github.com/lasigeBioTM/XMR4EL',
    python_requires=">3.12, <=3.13",
)