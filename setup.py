import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agent-maker",
    version="0.1.0",
    author="Agent Builder Team",
    author_email="example@example.com",
    description="A tool for generating LangGraph agents from natural language descriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/agent-maker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langgraph>=0.0.20",
        "openai>=1.3.0",
        "pydantic>=2.0.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "tweepy>=4.14.0",
        "argparse>=1.4.0",
        "jsonschema>=4.19.0",
    ],
    entry_points={
        "console_scripts": [
            "agent-maker=builder_agent:main",
        ],
    },
) 