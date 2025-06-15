import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]

setuptools.setup(
    name="uk-midas-client",                    
    use_scm_version=True,
    setup_requires=["setuptools-scm"],                                                      
    author="Katherine Whitelock",
    author_email="ktwhitelock@outlook.com",
    description="Prototype tools for analyzing Network Rail incident data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/katielocks/NetRailPipeline",
    package_dir={"": "src"},
    packages= setuptools.find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",                   
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
)
