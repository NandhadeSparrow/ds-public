## Predicting Customer Behavior in DVD Rental Using Deep Learning with AWS deployment

Predict customer behavior in a DVD rental business, specifically focusing on customer churn, movie genre preferences, and rental demand forecasting. The objective is to use deep learning techniques to create models that can assist in decision-making, marketing strategies, and personalized customer recommendations.


## Table of Contents
- [Predicting Customer Behavior in DVD Rental Using Deep Learning with AWS deployment](#predicting-customer-behavior-in-dvd-rental-using-deep-learning-with-aws-deployment)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
  - [Softwares needed](#softwares-needed)
  - [Code](#code)
  - [Python packages](#python-packages)
  - [Sparrowpy packages](#sparrowpy-packages)
  - [Environment\_variables](#environment_variables)
  - [Database Setup](#database-setup)
  - [Run App](#run-app)
- [Workflow](#workflow)
- [Contact](#contact)

## Setup
### Softwares needed
1. Jupyter Notebook     / Any Notebook IDE
2. Python 3.12          / venv with python=3.12
3. Git (with git bash)  / Download repo
4. AWS CLI
5. Docker

### Code

- Download project folder or full repo
- cd into the folder
``` bash
cd ds-public/3_dl/dvd_rental
```


### Python packages

Install all necessary packages
``` bash
pip install -r requirements.txt
```



### Sparrowpy packages

If you want to update my sparrowpy module
``` bash
cd utils/sparrowpy
git pull
cd ../..
```


### Environment_variables
Creating ```.env``` file using template
``` bash
cp .env_template .env
```

### Database Setup

Create database and table in PostgreSQL and add copy its credentials in ```.env``` file.

### Run App
``` bash
streamlit run app.py
```


## Workflow
[Slides]()

[Demo Video]()


## Contact
[LinkedIn](https://www.linkedin.com/in/nandhadesparrow)

---
^ [Back to table of contents](#table-of-contents)