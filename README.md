# AL_EXAmine
Alexa Crawler

#SET UP
First run virtualenv to virtualize your python environment. (DO THIS BEFORE MAKING THIS PROJECT THE ACTIVE DIRECTORY)

	virtualenv ENV
	source bin/activate
	pip install -r requirements.txt

Now you should have all of the necessary packages to run the project

#RUNNING THE CRAWLER

First `cd CrAwLEXA` then `scrapy crawl alexa_top_1m`, the output will be the file `content.bin` which is serialized in a umsgpack format. Good luck finding a better format.

To read objects from this file follow [these instructions](https://github.com/vsergeev/u-msgpack-python "u-msgpack-python github").

the format of the serialized objects is  {url:content}
