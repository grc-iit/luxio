import setuptools

setuptools.setup(
    name="luxio",
    packages=setuptools.find_packages(),
    version="0.0.1",
    author="Keith Bateman",
    author_email="kbateman@hawk.iit.edu",
    description="A tool for automatically configuring storage for distributed applications",
    url="https://github.com/scs-lab/luxio",
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 0 - Pre-Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: None",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Distributed Job Scheduling",
    ],
    long_description=""
)