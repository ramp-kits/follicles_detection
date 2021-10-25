import os
import glob
import xml.etree.ElementTree as ET

import click
import pandas as pd


def convert_class_to_int_label(label_text):
    conversion = {
        "Negative": 0,
        "Primordial": 1,
        "Primary": 2,
        "Secondary": 3,
        "Tertiary": 4,
    }
    return conversion[label_text]


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            label_text = member[0].text
            label_int = convert_class_to_int_label(label_text)

            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                label_text,
                label_int,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )

            xml_list.append(value)

    column_names = [
        "filename",
        "width",
        "height",
        "class",
        "label",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    return xml_df


@click.command()
@click.argument("xml_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("csv_path", type=click.Path(dir_okay=False))
def main(xml_folder, csv_path):
    """Convert xml files in folder XML_FOLDER to a single labels.csv"""
    parent_folder = os.path.dirname(xml_folder)
    assert csv_path.endswith(".csv")
    # output_filename = os.path.join(parent_folder, "labels.csv")

    # image_path = os.path.join(os.getcwd(), folder)
    print(f"Reading xml files in {parent_folder}")

    xml_df = xml_to_csv(xml_folder)
    xml_df.to_csv(csv_path, index=None)

    print(f"Successfully converted xml to csv: {csv_path}")


if __name__ == "__main__":
    main()
