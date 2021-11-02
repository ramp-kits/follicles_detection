# Detection of ovarian follicles


Authors : François Caud, Benjamin Habert and Alexandre Gramfort (Université Paris-Saclay)


## Getting started

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


## Notes for development


### repository structure

Folders `data`, `hidden_data` and `models` are not committed

```
% tree -L 3 .
.
├── data
│   ├── coupes_jpg
│   │   ├── D-1M01-2.jpg
│   │   ├── D-1M01-3.jpg
│   │   ...
│   ├── coupes_tiff
│   │   ├── D-1M01-2.tiff
│   │   ├── D-1M01-3.tiff
│   │   ...
│   ├── labels.csv
│   ├── scenario1
│   │   ├── Xml
├── hidden_data
│   ├── coupes_jpg
│   │   ├── D-1M07-1.jpg
│   │   ├── D-1M07-2.jpg
│   │   ├── D-1M07-3.jpg
│   │   ├── D-1M07-4.jpg
│   │   ├── D-1M07-5.jpg
│   │   └── D-1M07-6.jpg
│   └── labels.csv
├── models
│   └── classifier
│       ├── assets
│       ├── keras_metadata.pb
│       ├── saved_model.pb
│       └── variables
├── preprocessing_scripts
│   ├── reserve_hidden_data.py
│   ├── tiff_to_jpg.sh
│   └── xml_to_csv.py
├── requirements.txt
```

### Pre-processing raw data


- 1. Convert source .tiff images to lighter .jpg

    - place source images in `data/coupes_tiff/`
    - install [imagemagick](https://imagemagick.org/index.php)
    - run convert script

        ```
        $ ./.preprocessing_scripts/tiff_to_jpg.sh
        Converting images from data/coupes_tiff to data/coupes_jpg
        Converting data/coupes_tiff/D-1M01-2.tiff
        ```

      This process takes a few minutes. Images are output in `data/coupes_jpg/`

- 2. Convert xml files of labels and bounding boxes to a single `labels.csv`

    ```
    $ python preprocessing_scripts/xml_to_csv.py data/scenario1/Xml data/labels.csv
    Reading xml files in data/scenario1
    Successfully converted xml to csv: data/labels.csv
    ```

- 3. Set aside "hidden" data that will not be seen by challengers

    ```
    % python preprocessing_scripts/reserve_hidden_data.py
    ```

### Building a train dataset for a classification model

See notebook[`crop_images.ipynb`](crop_images.ipynb). This notebook takes
as input the image files in `data/images_jpg/` and the annotated
bounding boxes (`data/labels.csv`) to generate image thumbnails
organized as follows:

```
./data/split
├── test
│   ├── 0_Negative
│   ├── 1_Primordial
│   ├── 2_Primary
│   ├── 3_Secondary
│   └── 4_Tertiary
├── train
│   ├── 0_Negative
│   ├── 1_Primordial
│   ├── 2_Primary
│   ├── 3_Secondary
│   └── 4_Tertiary
└── val
    ├── 0_Negative
    ├── 1_Primordial
    ├── 2_Primary
    ├── 3_Secondary
    └── 4_Tertiary
```
