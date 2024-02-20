from setuptools import find_packages,setup

HYPEN_E_DOT = '-e .'
def get_requirements(file_path):
    '''
    This function will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if (HYPEN_E_DOT in requirements):
            requirements.remove(HYPEN_E_DOT)



    return requirements

setup(
name = 'MLproject',
version = '0.0.1',
author = 'Aniket Phadnis',
author_email = 'aniketphadnis6@gmail.com',
packages = find_packages(),
install_requires = get_requirements ('requirements.txt')
)
