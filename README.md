# AL_EXAmine
Alexa Crawler

#SET UP
First run virtualenv to virtualize your python environment. (DO THIS BEFORE MAKING THIS PROJECT THE ACTIVE DIRECTORY)

	virtualenv ENV
	source bin/activate
	pip install -r requirements.txt

Now you should have all of the necessary packages to run the project

#RUNNING THE CRAWLER

First `cd CrAwLEXA` then `scrapy crawl alexa_top_1m`, the output will be the folder `content/` which contains folders for each domain name and timestamped files for each url.

urlDiff takes the diffs between all files matching the same domain name and url between timestamps and deletes all but the lates file.

APILabels builds files containing the results of api queries on known domains.

run.sh does everything on a schedule and repeats. Multiple instances of scrapy will be launch effectively refreshing each url every 4 hours.

#LEARN

This section addresses the programs used for machine learning. Both program works in three parts: 

1. Feature Extraction 
2. Vector Creation 
3. Classification

## LEARN SECTORS

`learn_sectors.py` should be used for identifying the different sectors. run `learn_sectors.py -h` for help

## LEARN MALICIOUS PATTERNS
`learn_mal.py` is used to determin which urls are likely to be malicious. run `learn_mal.py -h` for help

# HELPER FUNCTIONS

We developed several helper files over the course of this project to automate/make simpiler certain executioins of code.

1. `data_classifier.py`- helper code that extracted out the time stamps for all of the URLs that were craweled. This output of data was used to compare the blacklist of domains against the URLS craweled and when. Primarily this was done to determine if 'when' we crawlwed the URLs if they appeared on any of the blacklist data we had. 
