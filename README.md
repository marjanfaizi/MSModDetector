# MSModDetector
A Python module to study post-translational modfication (PTM) patterns from individual ion mass spectrometry (I2MS) data.
I2MS is a top-down mass spectrometry (MS) approach that generates true mass spectra without the need to infer charge states.
MSModDetector first detects and quantifies mass shifts for a protein of interest and subsequently infers potential PTM patterns using linear programming. 


## Installation and requirements
MSModDetector can be installed with the following command:
```bash
$ git clone https://github.com/marjanfaizi/MSModDetector.git
```

Required Python packages to run this code will be installed with the following command:
```bash
$ cd MSModDetector
$ pip install -r requirements.txt
```

This repository was tested with Python 3.8.5 on macOS 12.2 and ...


## How to run MSModDetector
Make sure that the required Python packages are installed prior to the command line run
```bash
$ python main.py
```

## Configuration file
The file config.py has to be adjusted in order to refer to the correct input files. The configuration file requires the path name of the raw spectrum data, fasts file of the protein of interest, and the file name of the table of modifications considered for the PTM pattern analysis. Furthermore, names of the different replicates and conditions need to be listed. Additionally, meta parameters can be adusted as well in config.py that will impact the performance of mass shift detection and PTM patterm inference. Please see config.py for more detailed information about the meta parameters.
