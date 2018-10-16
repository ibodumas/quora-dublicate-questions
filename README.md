# quora-dublicate-questions
This project involves analyzing, and building several ML and DL models. The target data set is based on question pairs posted by Quora.

[First Quora Dataset Release: Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

# Model
The model adopted is MaLSTM with Google pre-trained word to vector.


# Deployment
Flask RESTful API with OpenAPI 2.0 specifications was developed to interface the pre-trained for testing new pairs of questions.
The API allows HTTP GET, with question1 and questions as query string.

Parameters: <br>
question1 and question2

Response Body <br>
{<br>
  "is_duplicate": false, <br>
  "probability": "0.37658742" <br>
}<br>

![alt text](https://github.com/ibodumas/quora-dublicate-questions/tree/master/images/RESTful.png)

# Installation Instruction
1. Pipenv is a tool that aims to bring the best of all packaging worlds to the Python world. 
2. For ease of installation, have pipenv installed.
3. Run "pipenv shell" from the project root directory, this will create a VIRTUAL ENVIRONMENT and install all the project dependencies in it. 
4. Make sure that each sub-folders in the project is in the sys.path. 
5. Run model_train.py to train the model. This will also save the best model to file.
6. Run wsgi.py to initiate the RESTfull API. Default port is 8080. OR run with gunicorn --bind 0.0.0.0:8080 wsgi:API

Thank You.

# Download the Google news word2vec pretrained model from terminal

1: wget --save-cookies cookies.txt --keep-session-cookies <br>
--no-check-certificate 'https://docs.google.com/uc? <br>
export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O- | <br>
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'

This will give you an ID, say
Code: Se3r

2: wget --load-cookies cookies.txt 'https://docs.google.com/ <br>
uc?export=download&confirm=YOURCODEID&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O <br>
GoogleNews-vectors-negative300.bin.gz


Kindly replace the ID with yours.
