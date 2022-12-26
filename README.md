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
- fasta file of the protein of interest stored in the directory "fasta_files"
- a csv table with all the PTMs that should be considered for inferring PTM patterns stored in the directory "modifications"
- a csv table, called metadata.csv, containing information about the experimetnal set up stored in the directory "raw_data"

### Fasta file
The fasta file containing the sequnce of the protein of interest can be downloaded from https://www.uniprot.org or any other protein sequence database.

### Modification table
The modification table should contain the following columns:
- unimod_id: The Unimod identifier of the PTM	
- ptm_id:	Abbreviation of the PTM, which is used to display the PTM patterns
- ptm_name: Full name of the PTM
- ptm_mass: Average mass in Da of the PTM
- composition: Elemental composition of the PTM
- upper_bound: Determines the upper bound of modification sites for the PTM
- site: Determined the amino acid resiudes that can be modified	
- function: Is the PTM biological or an artifact

An modification table example is given in the directory "modifications". 

### Metadata
The metadata file should contain informtation about the file names of the raw data, their condition and the repkicate number. See in the "raw_data" directory for an example table.

## How to run MSModDetector
Make sure that the required Python packages are installed prior to the command line run
```bash
$ cd src
$ python main.py
```

