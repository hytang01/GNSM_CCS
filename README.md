GNSM for CCS Optimization Example
This repository provides an example implementation of the Graph Neural Simulation Model (GNSM) applied to Carbon Capture and Storage (CCS) optimization. The code demonstrates the inference and optimization process used in our study.

Overview
Purpose:
This example illustrates how GNSM can be utilized for optimizing CCS processes.

Core Module:
The central module is feval.py, which serves as the main entry point. It calls various supporting functions and modules necessary to run the evaluation and optimization routines.

Repository Structure
bash
Copy
├── feval.py                # Main evaluation module
├── [Other modules].py      # Supporting modules called by feval.py
├── README.md               # This file
└── environment.txt         # (Optional) Full conda environment package list
Installation and Setup
Conda Environment
To reproduce the results, you need to set up a conda environment with the required dependencies. The environment used for this project includes (but is not limited to) the following packages:

yaml
Copy
_libgcc_mutex             0.1                        main
_openmp_mutex             5.1                       1_gnu
anyio                     3.5.0                    pypi_0    pypi
argon2-cffi               21.3.0             pyhd3eb1b0_0
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
asttokens                 2.0.5              pyhd3eb1b0_0
attrs                     22.1.0                   pypi_0    pypi
backcall                  0.2.0              pyhd3eb1b0_0
beautifulsoup4            4.11.1                   pypi_0    pypi
blas                      1.0                         mkl
bleach                    4.1.0              pyhd3eb1b0_0
bottleneck                1.3.5                    pypi_0    pypi
brotli                    1.0.9                h5eee18b_7
brotli-bin                1.0.9                h5eee18b_7
brotlipy                  0.7.0                    pypi_0    pypi
bzip2                     1.0.8                h7b6447c_0
ca-certificates           2024.8.30            hbcca054_0    conda-forge
certifi                   2024.8.30          pyhd8ed1ab_0    conda-forge
cffi                      1.15.1                   pypi_0    pypi
charset-normalizer        2.0.4              pyhd3eb1b0_0
comm                      0.1.2                    pypi_0    pypi
contourpy                 1.0.5                    pypi_0    pypi
cryptography              38.0.4                   pypi_0    pypi
cudatoolkit               11.3.1               h2bc3f7f_2
cudnn                     8.2.1                cuda11.3_0
cupti                     11.3.1                        0
cycler                    0.11.0             pyhd3eb1b0_0
dbus                      1.13.18              hb2f20db_0
deap                      1.4.1                    pypi_0    pypi
debugpy                   1.5.1                    pypi_0    pypi
decorator                 5.1.1              pyhd3eb1b0_0
defusedxml                0.7.1              pyhd3eb1b0_0
entrypoints               0.4                      pypi_0    pypi
et_xmlfile                2.0.0              pyhd8ed1ab_0    conda-forge
executing                 0.8.3              pyhd3eb1b0_0
expat                     2.4.9                h6a678d5_0
fastjsonschema            2.16.2                   pypi_0    pypi
fftw                      3.3.9                h27cfd23_1
flit-core                 3.6.0              pyhd3eb1b0_0
fontconfig                2.14.1               h52c9d5c_1
fonttools                 4.25.0             pyhd3eb1b0_0
freetype                  2.12.1               h4a9f257_0
future                    0.18.2                   pypi_0    pypi
giflib                    5.2.1                h5eee18b_1
glib                      2.69.1               he621ea3_2
gst-plugins-base          1.14.0               h8213a91_2
gstreamer                 1.14.0               h28cd5cc_2
h5py                      3.7.0                    pypi_0    pypi
hdf5                      1.10.6               h3ffc7dd_1
icu                       58.2                 he6710b0_3
idna                      3.4                      pypi_0    pypi
intel-openmp              2021.4.0          h06a4308_3561
ipykernel                 6.19.2                   pypi_0    pypi
ipython                   8.8.0                    pypi_0    pypi
ipython_genutils          0.2.0              pyhd3eb1b0_1
jedi                      0.18.1                   pypi_0    pypi
jinja2                    3.1.2                    pypi_0    pypi
joblib                    1.1.1                    pypi_0    pypi
jpeg                      9e                   h7f8727e_0
jsonschema                4.16.0                   pypi_0    pypi
jupyter-client            7.4.8                    pypi_0    pypi
jupyter-core              5.1.1                    pypi_0    pypi
jupyter-server            1.23.4                   pypi_0    pypi
...
[zstd, etc.]
Note: For brevity, only a portion of the complete package list is shown above. The full list is available in the environment.txt file provided in the repository.

Creating the Environment
You can create a new conda environment and install the required packages manually, or use an environment file if one is provided. To create the environment manually:

bash
Copy
conda create --name ccs_gnsm_env python=3.10.9
conda activate ccs_gnsm_env
Then, install the required packages using conda install and pip install as necessary, referring to the package list above.

Running the Code
After setting up the environment, you can run the main module with:

bash
Copy
python feval.py
Please refer to the comments within feval.py and other modules for further instructions and configuration options.

Additional Notes
This repository represents a proof-of-concept for applying GNSM in the context of CCS optimization.
Future developments may include additional model training modules and extended functionality for business applications.
Only the inference and optimization code has been provided in this repository to allow reproducibility of the key results while preserving proprietary training methods.
