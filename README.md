# KubAnomaly-SFSOINN
KubAnomaly-SFSOINN
This project implements an unsupervised anomaly detection model for Docker containers using the Soft-Forgetting Self-Organizing Incremental Neural Network (SF-SOINN) algorithm within the KubAnomaly framework. The goal is to detect complex and evolving security threats in containerized environments, leveraging unsupervised learning.

REQUIREMENTS:-----------------------------------
* Python: Version 3.11 or above is required.
* Required Libraries: Install the necessary packages using the following command:

  pip install -r requirements.txt

Files and Folder Structure:-----------------------------------
* KubAnomaly-SFSOINN.py: The main Python script for running the KubAnomaly-SFSOINN anomaly detection model.
* SF_SOINN.py: Ensure this file is in the same directory as KubAnomaly-SFSOINN.py. It contains the implementation of the SF-SOINN algorithm.
* Data: Folder containing CSV files for the KubAnomaly experiment dataset. Do not modify or delete any files within this folder. The data can be obtained from the KubAnomaly GitHub repository.

Data Credits:-----------------------------------
The data used for this project is sourced from the KubAnomaly DataSet by GitHub user a18499. Please refer to their repository for additional information on dataset structure and content.

Installation and Setup:-----------------------------------
Clone the Repository:

git clone https://github.com/Dencotexts/KubAnomaly-SFSOINN.git

cd KubAnomaly-SFSOINN


cd KubAnomaly-SFSOINN

Install Python 3.11 or Above: Make sure Python 3.11 or a later version is installed on your system. You can download it from python.org.

Install the Required Libraries: Run the following command to install all required libraries listed in the requirements.txt file:

pip install -r requirements.txt

Ensure the Presence of SF_SOINN.py: Confirm that the SF_SOINN.py file is located in the same directory as KubAnomaly-SFSOINN.py.

Prepare the Data Folder: Download the Data folder from the KubAnomaly DataSet GitHub link and place it in the project directory. Do not delete or modify any files within this folder after downloading.

Running the Model:-----------------------------------
Once the setup is complete, you can run the anomaly detection model as follows:

python KubAnomaly-SFSOINN.py
Make sure you are in the project directory where KubAnomaly-SFSOINN.py, SF_SOINN.py, and the Data folder are located. This will execute the KubAnomaly-SFSOINN model and begin the anomaly detection process.

Notes
This project is intended for educational and research purposes.
Please cite the original KubAnomaly DataSet repository if you use this project in any publications.
