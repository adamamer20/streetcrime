from data_processing_raw import data_processing
import os

def main():
    directory = os.path.dirname(__file__)
    for _ in range(2):
        parent_directory = os.path.split(directory)[0]
        directory = parent_directory
    data_processing(directory)    
    
if __name__ == "__main__":
    main()
    
