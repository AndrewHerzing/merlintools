from setuptools import setup

setup(
    name='MerlinTools',
    version='0.1dev',
    packages=['merlintools',],
    description='Handle 4D-STEM data collected with a Merlin detector.',
    long_description=open('README.txt').read(),
    install_requires = [
            'numpy',
            'hyperspy',
            'fpd',
            'pyxem',
            ],
    author = 'Andrew Herzing',
    author_email = 'andrew.herzing@nist.gov',
    license = 'GPL v3',
    keywords = [
        'STEM',
        'data analysis',
        'microscopy',
        '4D-STEM',
        ],
)