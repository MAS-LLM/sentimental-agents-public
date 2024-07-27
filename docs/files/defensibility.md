# Defensibility Check

This code performs a defensibility check on folders in a specified directory. The defensibility check evaluates whether the arguments made by agents are backed by evidence that can be found and referenced from the candidates' CV.

## Logic

1. Load resume samples from a CSV file.
2. Define a function `load_resume` to load resume data from text. This function creates a temporary file, writes the resume text to the file, and then loads the data using the `UnstructuredReader` class.
3. Define a function `run_defensibility_check` to run the defensibility check on folders in a directory.
   - Get all the folders in the specified directory.
   - Sort the folders based on modification time.
   - Iterate over each folder:
     - Extract the candidate name and resume text from the corresponding row in the resume samples.
     - Load the simulation data from the JSON file in the folder.
     - Load the resume documents using the `load_resume` function.
     - Create a `SimpleNodeParser` and a `ServiceContext` for indexing the documents.
     - Create a `VectorStoreIndex` from the documents and the service context.
     - Iterate over each agent's messages in the simulation data:
       - Retrieve the response from the index using the message text as a query.
       - If the response is empty, add an entry to the defensibility list with a score of 0.
       - If the response is not empty, add an entry to the defensibility list with the source text and the score from the response.
     - Create a DataFrame from the defensibility list.
     - Append the DataFrame to a list of DataFrames.
     - Append the candidate name to a list of sheet names.
   - Write the DataFrames to an Excel file with sheet names corresponding to the candidate names.
   - Return the list of DataFrames.

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Prepare the resume samples CSV file with the following columns: `resume`.
3. Place the simulation data JSON files in separate folders within the specified directory.
4. Run the `run_defensibility_check` function to perform the defensibility check on the folders.
5. The defensibility scores will be saved in an Excel file named `defensibility_scores.xlsx` in the output directory.

Note: Make sure to set the correct paths for the resume samples CSV file and the output directory in the code.

## Dependencies

- pandas
- tempfile
- dotenv
- pathlib
- llama_hub
- llama_index
- openpyxl