# Getting started


Authors : François Caud, Benjamin Habert and Alexandre Gramfort (Université Paris-Saclay)


## Install

```
git clone <https repo>
cd follicles_detection/
```


To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. We recommand installing these dependencies in a
specific python environment.

#### Installing dependencies in a `virtualenv`

You can install install the dependencies with the
following command-line:

```bash
# create a local virtualenv and activate it
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Installing dependencies in a `conda` environment

TODO

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

## Download data

```
python download_data.py
```

This will create the following folders and files:

```
tree -L 2 data   
data
├── test
│   ├── D-1M06-1.jpg
│   ..
│   ├── D-1M06-5.jpg
│   └── labels.csv   <-- bounding boxes and labels for test images
└── train
    ├── D-1M01-2.jpg
    ...
    ├── D-1M05-6.jpg
    └── labels.csv   <-- bounding boxes and labels for train images
```

## Check installation

```
ramp-test --submission starting_kit
ramp-test --submission random_classifier --quick-test
```

## Build documentation

```
cd doc
make html
```

Open the file `doc/build/html/index.html` in a browser.

## Challenge description

Get started with the [dedicated notebook]


## Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

## To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)

