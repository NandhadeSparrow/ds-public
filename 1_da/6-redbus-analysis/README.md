# Redbus Scraping and Filtering

## Description
A Streamlit application that can be used to analyse redbus data.

## Table of Contents
- [Redbus Scraping and Filtering](#redbus-scraping-and-filtering)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Softwares needed](#softwares-needed)
    - [Code](#code)
    - [Python packages](#python-packages)
    - [Environment\_variables](#environment_variables)
    - [Database Setup](#database-setup)
    - [Run App](#run-app)
  - [Workflow](#workflow)
  - [Contact](#contact)
## Setup
### Softwares needed
1. IDE (VS Code)
2. Python
3. Git (with git bash)
4. PostgreSQL

### Code

Clone this repository and ```cd``` into that directory
``` 
git clone https://github.com/NandhadeSparrow/ds-redbus-analysis.git 
cd ds-redbus-analysis
```


### Python packages

Install all necessary packages
``` 
pip install -r requirements.txt
```

### Environment_variables
Creating ```.env``` file using template
``` 
cp env_template.txt .env
```

### Database Setup

Create a app and collection in MongoDB Atlas and add its credentials in ```.env``` file.

Create a local sql database and add its credentials in ```.env``` file

### Run App
``` 
streamlit run Intro.py
```


## Workflow
[Slides]()

[Demo Video]()


## Contact
[LinkedIn](https://www.linkedin.com/in/nandhadesparrow)

---
^ [Back to table of contents](#table-of-contents)
