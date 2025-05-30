import os
import shutil
import zipfile


def unzip_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                unzip_directory = os.path.splitext(file_path)[0]
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_directory)
                for extracted_root, extracted_dirs, extracted_files in os.walk(unzip_directory):
                    for extracted_file in extracted_files:
                        shutil.move(os.path.join(extracted_root, extracted_file),
                                    os.path.join(directory, extracted_file))
                shutil.rmtree(unzip_directory)


if len(list(filter(lambda x: ".pkl" in x, os.listdir('data/')))) < 3:
    print("Unzipping data")
    directory_to_unzip = 'data/'
    unzip_all_files(directory_to_unzip)

print("Data loaded")
