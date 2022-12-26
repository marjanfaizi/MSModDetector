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

This repository was tested with Python 3.8.5 on macOS 12.2.

## Required files
In order to run MSModDetector on your raw I2MS data to identify mass shifts for a protein of interest and infer potential PTM patterns, the following files are rquired:
- fasta file of the protein of interest stored in the fasta_files directory
- a csv table with all the PTMs that should be considered for inferring PTM patterns
- a csv table, called metadata.csv, containing information about the experimetnal set up stored in the raw_data directory


## How to run MSModDetector
Make sure that the required Python packages are installed prior to the command line run
```bash
$ cd src
$ python main.py
```

