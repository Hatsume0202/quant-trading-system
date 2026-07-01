from setuptools import setup, find_packages

setup(
    name="quant-trading-system",
    version="0.1.0",
    description="A quantitative stock trading system with backtesting and simulation",
    author="Quant Trader",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.38",
        "pandas>=2.2.0",
        "numpy>=2.0.0",
        "matplotlib>=3.9.0",
    ],
    entry_points={
        "console_scripts": [
            "quant-trading=main:main",
        ],
    },
)
