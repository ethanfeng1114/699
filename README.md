Install environment
conda create -n 699 python=3.11 
^Done

Activate environment:
conda activate 699

Dependencies
conda install -c conda-forge numpy scipy pandas matplotlib jupyterlab -y
conda install -c conda-forge ipywidgets
pip install pybamm 

do not install pybamm with conda 


export PYBAMM_DISABLE_TELEMETRY=1