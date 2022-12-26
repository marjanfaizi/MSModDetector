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
In order to run MSModDetector on your raw I2MS data to identify mass shifts for a protein of interest and to infer potential PTM patterns, the following files are rquired:
- raw I2MS data should be stored in the directory "raw_data"
- a csv table, called metadata.csv, containing information about the experimental set up stored in the directory "raw_data"
- fasta file of the protein of interest stored in the directory "fasta_files"
- a csv table with all the PTMs that should be considered for inferring PTM patterns stored in the directory "modifications"

### Raw data and metadata table
The raw data and the metadata table should be stored in the "raw_data" directory. The metadata file should contain informtation about the file names of the raw data, their condition and the replicate number. See in the "raw_data" directory for an example table.

### Fasta file
The fasta file containing the sequence of the protein of interest can be downloaded from https://www.uniprot.org or any other protein sequence database.

### Modification table
The modification table should contain the following columns:
- unimod_id: The Unimod identifier of the PTM	from https://www.unimod.org
- ptm_id:	Abbreviation of the PTM, which is used to display the PTM patterns
- ptm_name: Full name of the PTM
- ptm_mass: Average mass in Da of the PTM
- composition: Elemental composition of the PTM
- upper_bound: Determines the upper bound of modification sites for the PTM
- site: Determines the amino acid resiudes that can be modified	
- function: Is the PTM function biological or is it an artifact

A modification table example is given in the directory "modifications". 

## How to run MSModDetector
Make sure that the required Python packages are installed and all required files are stored in the correct directories. To run MSModDetector you need to spcfiy the directory where the raw data and metadata table are stored, the name of the modification table and fasta file, the start mass and end mass of the range where the algorithm should search for mass shifts, and the size of the sliding window that iterates throughh the mass spectrum and searches for mass shifts. Here is an example how to run MSModDetector for experimetnal data of endogenous p53.

```bash
$ cd src
$ python main.py -data "../raw_data/" -mod "modifications_P04637.csv" -fasta "P04637.fasta" -start 43750.0 -end 44520.0 -wsize 10
```

Other meta parameters can be changed if the default values are not suited. A description of all meta parameters can be find using the help function:
```bash
python main.py --help
```

MSModDetector outputs a table with the identified mass shifts, the corresponding potential PTM patterns, and relative abundances for every mass shift. If you choose to obtain more than one possible PTM pattern solution for every mass shift, then another table with k optimal solutions will be generated as well. All results will be stored in "output". 

