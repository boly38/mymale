
## Installation (Windows host)

### step 0 ) Python 3.11.7 
why 3.11.7 ?
- cf. https://discuss.tensorflow.org/t/teachable-machine-h5-wont-load/23005/10+

setup python 3.11.7 installer (not embed!)
- from [python/download](https://www.python.org/downloads/) 
- customize installation path, ex : `C:\Tools\Python3117`

### step 1) Python winenv

```bash
# cd mymale
/C/Tools/Python3117/python.exe -m venv ./winvenv
# ./winvenv/Scripts/activate.bat
# or
source ./winvenv/Scripts/activate
# verify version
python --version
## expect 3.11.7
# NOT TO DO ## update pip  (ex. from 23.2.1 to 24)
# NOT TO DO ## python -m pip install --upgrade pip
```

### Step 2) Project dependencies
````bash
# cd mymale
# source ./winvenv/Scripts/activate
# install dependencies
python -m pip install -r ./requirements.txt
````
