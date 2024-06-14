# Source code of paper: ["Aid effectiveness in sustainable development: A multidimensional approach"](https://doi.org/10.1016/j.worlddev.2023.106256)

## By [Omar Guerrero](https://oguerr.com), Daniele Guariso, and Gonzalo Castañeda

All the scripts to pre-process the data and perform the analysis are contained in the folder `code`. The data is too large to be hosted in this repository, so we provide a zip file that you can download [here](https://www.dropbox.com/scl/fo/4kg6svoyo65qpimozekbl/AOiYdZOlndUrfm1otQ5jUo0?rlkey=0866hyu5mkb4df79bgvcvjkmp&dl=0). 
First, clone this repository.
Then, decompress the data zip file and place its content into the `data` folder of your copy of the repository. This will provide the folder structure to run every script.

The scripts are organised numerically to follow a specific execution order.
Scripts 00 to 25 are for preparing the data.
Scripts 30 onwards perform the simulations necessary for the analysis.
No scripts to produce the figures of the paper are provided.

While most scripts are in Python, script 22 is in R.
This script uses a library called [sparsebn](https://github.com/itsrainingdata/sparsebn).
Unfortunately, sparsebn is no longer supported, so you should contact its creator.
However, this module is not essential to run the model used in the paper, it only helps to create networks of co-movements between the indicators.
You can always estimate these networks with alternative approaches and [here a is paper](https://doi.org/10.1016/j.im.2020.103342) that points you in the right direction.
In any case the data file contains the networks that we estimated using sparsebn so you can replicate all the results in the paper.


