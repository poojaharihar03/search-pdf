# search-pdf

### Step-by-Step Code Execution Instructions:
  Clone the project

```bash
  git clone https://github.com/poojaharihar03/search-pdf.git
```
 #### Create a virtual envirnoment on windows
 1. Use the cd command to navigate to the directory where you want to create the virtual environment. For example:
```bash
  cd path/to/your/project
```
 2. Use the following command to create a virtual environment named venv:
```bash
  python -m venv venv
```
 3. To activate the virtual environment, navigate to the Scripts directory inside the venv folder and run the activate script. Use the following command:
```bash
  venv\Scripts\activate
```

#### Create a virtual envirnoment on mac
 1. Use the cd command to navigate to the directory where you want to create the virtual environment. For example:
```bash
  cd path/to/your/project
```
 2. Use the following command to create a virtual environment named venv:
```bash
  python3 -m venv venv
```
 3. To activate the virtual environment, navigate to the Scripts directory inside the venv folder and run the activate script. Use the following command:
```bash
  source venv\Scripts\activate
```
Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server
if you wish to check out the text generation
```bash
  streamlit run app.py
```
if you wish to check out the text to speech
```bash
  streamlit run stapp.py
```
---
 4. To deactivate the virtual environment,  Use the following command:
```bash
  deactivate
``` 
