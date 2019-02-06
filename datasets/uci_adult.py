from datasets import sourceforge_data_processor as processor
import numpy as np

zip_data_path = "data/adult.dat"
zip_domain_path = "data/adult.domain"


def data(delete_file: bool = False):
    processor.download_data()
    processor.extract_data(zip_data_path, zip_domain_path)
    (x, y) = processor.process_data(zip_data_path, zip_domain_path)
    if delete_file:
        processor.delete_files(["data/"])

    return np.asarray(x), np.asarray(y)
