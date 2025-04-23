from os import getenv
from dotenv import load_dotenv
from cleanvision import Imagelab

load_dotenv()

# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(data_path=f"{ getenv( "DATASETS_PATH", "../datasets" ) }/combined_dataset")

# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues()

# Produce a neat report of the issues found in your dataset
imagelab.report()
