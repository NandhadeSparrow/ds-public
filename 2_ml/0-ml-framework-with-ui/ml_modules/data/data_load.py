# load data
selectinput = 'csv' # sns / csv / kaggle

inputcsv = 'Unicorn_Companies.csv'
inputsns = 'taxis'
# inputkaggle = 'data/fashion-mnist_train.csv'

if selectinput == 'csv':
  df = pd.read_csv(inputcsv)
elif selectinput == 'sns':
  df = sns.load_dataset(inputsns)
# elif selectinput == 'kaggle'
#   !kaggle datasets download -d zalando-research/fashionmnist -p data
#   !unzip data/fashionmnist.zip -d data  # Unzip if necessary
#   os.environ['KAGGLE_USERNAME'] = 'Your_Kaggle_Username'
#   os.environ['KAGGLE_KEY'] = 'Your_Kaggle_API_Key'
#   df = pd.read_csv(inputkaggle)  # Assuming the filename



# From SQL
import sqlite3
conn = sqlite3.connect('database.db')
data = pd.read_sql_query('SELECT * FROM table_name', conn)

Data Selection:
Select specific columns or rows from your DataFrame.

data['column_name']
data.loc[data['condition']]


  if source_type == 'csv':
      data = pd.read_csv(source)
  elif source_type == 'json':
      with open(source, 'r') as file:
          data = json.load(file)
      data = pd.DataFrame(data)  # Convert JSON to DataFrame
  elif source_type == 'api':
      response = requests.get(source)
      data = response.json()
      data = pd.DataFrame(data)  # Convert JSON response to DataFrame
  else:
      raise ValueError("Unsupported source type. Supported types: 'csv', 'json', 'api'")

  return data

