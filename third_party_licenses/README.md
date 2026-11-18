# Third-party licenses

This project is released under the Apache License 2.0 (see [`../LICENSE`](../LICENSE)).
It depends on the packages below, which are not included in this repository: `pixi install` downloads them from [conda-forge](https://conda-forge.org) at the versions pinned in [`../pixi.lock`](../pixi.lock).
Each is distributed under its own license, whose full text is in this folder, taken from the installed package.

| Package | Version used | License | License text | Project |
| --- | --- | --- | --- | --- |
| Python | 3.15.2 (Linux), 3.14.7 (macOS) | PSF License (Python-2.0) | [`python.txt`](python.txt) | https://www.python.org |
| NumPy | 2.5.4 (Linux), 2.5.3 (macOS) | BSD-3-Clause | [`numpy.txt`](numpy.txt) | https://numpy.org |
| Matplotlib (`matplotlib-base`) | 3.11.2 | Matplotlib License (PSF-based) | [`matplotlib.txt`](matplotlib.txt) | https://matplotlib.org |
| ipykernel | 6.31.0 | BSD-3-Clause | [`ipykernel.txt`](ipykernel.txt) | https://github.com/ipython/ipykernel |
| jupyter_client | 8.10.0 | BSD-3-Clause | [`jupyter_client.txt`](jupyter_client.txt) | https://github.com/jupyter/jupyter_client |
| pytest | 9.2.0 (Linux), 9.1.1 (macOS) | MIT | [`pytest.txt`](pytest.txt) | https://pytest.org |
| pip | 26.2.2 (Linux), 26.2.1 (macOS) | MIT | [`pip.txt`](pip.txt) | https://pip.pypa.io |

The code of this project imports only NumPy and Matplotlib, besides the Python standard library.
ipykernel and jupyter_client run the notebook, pytest runs the tests, and pip is installed as a tool.

NumPy and pip also bundle smaller components under their own licenses; those are listed in each package's own distribution (its `.dist-info/licenses` folder once installed).
These packages are used unmodified.
