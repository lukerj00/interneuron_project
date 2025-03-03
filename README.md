# Interneuron Project

This repo contains parameter inference, simulation and analysis code for Phase Space Structure of Interneuronal Circuits (https://www.overleaf.com/project/61f2c0d42959e1d6c8dc0f78), based on mouse V1 data from Adesnik Lab.

## Project Structure
- `data/`: Contains experimental data and parameter files
  - `Data_cell_types_small_size.mat`: raw V1 data for 4 neuron types across 6 contrast values
  - `fits_cnd1_top200_250102_125616.pkl`: 200 example fits, using parameter_inference.ipynb
  - `Parameters-DataSet-1-*.mat`: 20 example fits from previous project work
- `figures/`: output dir for generated figures
- `notebooks/`: notebooks for fitting, simulation and analysis
  - `01_parameter_inference.ipynb` see below
  - `02_simulations.ipynb` see below
  - `03_analysis.ipynb` see below
- `scripts/`: python modules with core functionality
  - `analytics_functions.py`: Data processing and statistical analysis
  - `fitting_functions.py`: Model fitting and parameter optimization
  - `plotting_functions.py`: Visualization and figure generation
  - `simulation_functions.py`: Running computational simulations

## Installation

```bash
python -m venv venv_ip
source venv_ip/bin/activate 
pip install -r requirements.txt
```

## Usage

### Parameter Inference
`01_parameter_inference.ipynb` contains a step-by-step procedure for inferring parameters (using NNLS) and mean data fits (using PNAS polynomial method) from the data in `/data/Data_cell_types_small_size.mat`, based on a number of options (fitting assumptions, priors, etc). 

The second part of the notebook visualises the data and fits.

### Simulations
`02_simulations.ipynb` contains simple euler scheme simulations for the data fits using a first order model of the 4 population means. Options exist for sim of global/linearised system, with optional white/coloured (OU) noise in linearised case.

Notebook also contains analytics of correlations between populations in latter case of linearised system is driven by noise.

Final part features phase space simulations of system, projected along 2 of 4 populations.

### Analytics
`03_analysis.ipynb` contains various analyses of the first order model. Most analysis feature 'heatmap' plots of the effect if driving the system by external input along any of the 4 populations, plotting resultant stability, rate, gain or variance of the (linearised) system.

Also contains visualisations of various analytics when system is driven by external inputs which are interpolated between an 'initial' and 'final' vector.

## License
open-source (MIT license)

## Contact
luke.johnston00@gmail.com

## Upcoming changes
-improved docs
-hopf analysis notebook
