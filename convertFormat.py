from utils.convert_annotations import convert
from utils.read_annotations import read_ra_labels_csv


if __name__ == "__main__":
    path = 'F:\\data\Automotive\\2019_04_09_cms1000'
    convert(path)
    res = read_ra_labels_csv(path)
    print(len(res))

