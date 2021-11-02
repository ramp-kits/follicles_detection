# Getting started


Authors : François Caud, Benjamin Habert and Alexandre Gramfort (Université Paris-Saclay)


### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Download data

TODO

### Check installation

```
ramp-test --submission starting_kit
ramp-test --submission random_classifier
```

### Build documentation

```
cd doc
make html
```

Open the file `doc/build/html/index.html` in a browser.

### Challenge description

Get started with the [dedicated notebook]


### Test a submission

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

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)

