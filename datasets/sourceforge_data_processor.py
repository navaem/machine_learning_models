from urllib import request
from zipfile import ZipFile
import os
import shutil

base_url = "https://sourceforge.net/"
path_uri = "projects/privbayes/files/latest/download"
zip_filename = "../privbayes-new.zip"


def download_data():
    """
    Downloads the dataset zip from sourceforge if the zipfile has not been previously downloaded.

    """
    if not os.path.isfile(zip_filename):
        req = request.Request(base_url + path_uri)
        response = request.urlopen(req)
        zip_file = response.read()
        zip_fh = open(zip_filename, "wb")
        zip_fh.write(zip_file)
        zip_fh.close()


def extract_data(zip_data_path: str, zip_domain_path: str):
    """
    Unzips the desired data path and domain path from the sourceforge zip file.

    :param zip_data_path: path within zip file to data
    :param zip_domain_path: path within zip file to domain
    """
    with ZipFile(zip_filename, "r") as zip:
        zip.extract(member=zip_data_path)
        zip.extract(member=zip_domain_path)


def process_data(zip_data_path: str, zip_domain_path: str) -> (list, list):
    """
    Processes the data into a usable list for machine learning

    :param zip_data_path: path to extracted data
    :param zip_domain_path: path to extracted domain
    :return: x_data and y_data
    """
    domain_fh = open(zip_domain_path, "r")
    domain = domain_fh.readlines()
    domain_cat = []
    for line in domain:
        if line[0] is "C":
            domain_cat.append(["C"])
        else:
            line = line.replace("\n", "").replace("}", "").replace("{", "")
            categories = line.split(" ")[1:]
            cat_dict = dict()
            for i in range(len(categories)):
                cat_dict[categories[i]] = i
            domain_cat.append(["D", cat_dict])

    data_fh = open(zip_data_path, "r")
    data_lines = data_fh.readlines()
    y_categories = domain_cat[-1][1]

    x_data = []
    y_data = []
    for line in data_lines:
        line = line.replace("\n", "").strip()
        line_data = line.split(" ")
        if len(line_data) == len(domain_cat):
            x_inst = []
            for i in range(len(line_data) - 1):
                if domain_cat[i][0] == "D":
                    x_inst.append(domain_cat[i][1][line_data[i]])
                else:
                    x_inst.append(int(line_data[i]))
            x_data.append(x_inst)
            y_data.append(y_categories[line_data[-1]])

    domain_fh.close()
    data_fh.close()

    return x_data, y_data


def delete_files(files: list):
    for file in files:
        if file == "/":
            raise OSError("Won't delete home dir")
        if os.path.isfile(file):
            os.remove(file)
        elif os.path.isdir(file):
            shutil.rmtree(file)
