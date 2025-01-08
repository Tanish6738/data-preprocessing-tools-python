import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import argparse

def main(file_paths):
    """
    Orchestrates the data cleaning and preprocessing process on a list of data files.
    """
    print("Loading datasets...")
    dataframes = load_datasets(file_paths)
    print("Datasets loaded successfully.")

    for i, df in enumerate(dataframes):
        print(f"Processing DataFrame {i + 1} ({file_paths[i]})...")  # Print the current working file
        dataframes[i] = handle_missing_values(df)
        dataframes[i] = handle_duplicates(dataframes[i])
        
        while True:
            column = input(f"Enter the column name for outlier detection in DataFrame {i + 1} (or 'q' to quit): ")
            if column.lower() == 'q':
                break
            if column not in df.columns:
                print(f"Column '{column}' not found in DataFrame. Please enter a valid column name.")
                continue
            dataframes[i] = handle_outliers(df, column, method='iqr')
            print(f"Outliers handled for column '{column}'.")

        user_wants_selection = input("\nDo you want to select features for this DataFrame (y/n)? ").lower()
        if user_wants_selection == 'y':
            dataframes[i] = select_features(df.copy())
            print("Feature selection completed.")

        dataframes[i] = handle_categorical_variables(df)
        print("Categorical variables handled.")

    common_columns = find_common_columns(dataframes)
    combined_df = combine_datasets(dataframes, common_columns)
    print("\nCombined DataFrame:")
    print(combined_df.head())

    filenames = []
    for i in range(len(dataframes)):
        filename = input(f"\nEnter a filename for DataFrame {i + 1} (without extension): ")
        filenames.append(filename + '.csv')
    save_datasets(dataframes, filenames)
    print("\nDatasets saved successfully.")

def load_datasets(file_paths):
    """
    Loads multiple datasets from the provided file paths and returns a list of pandas DataFrames.
    """
    dataframes = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)  # Assuming CSV files (modify for other formats)
            dataframes.append(df)
            print(f"Successfully loaded data from {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except pd.errors.ParserError:
            print(f"Error: Parsing error encountered for file - {file_path}")
    return dataframes

def find_common_columns(dataframes):
    """
    Finds common columns among all datasets.
    """
    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns = common_columns.intersection(df.columns)
    return list(common_columns)

def combine_datasets(dataframes, common_columns):
    """
    Combines datasets based on user choice.
    """
    print("Common columns in all datasets:")
    for col in common_columns:
        print(col)
    choice = input("Do you want to combine datasets based on a common column (y/n)? ").lower()
    if choice == 'y':
        common_column = input("Enter the common column name: ")
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        return combined_df
    else:
        combined_df = pd.concat(dataframes, axis=1)
        return combined_df

def handle_missing_values(df, strategy='mean'):
  """
  Handles missing values in a DataFrame using SimpleImputer.

  Args:
      df (pandas.DataFrame): The DataFrame containing missing values.
      strategy (str, optional): The current imputation strategy (default: 'mean').

  Returns:
      pandas.DataFrame: The DataFrame with missing values imputed or rows with missing values dropped based on user choice.

  Prints:
      Information about missing values, user input, and imputation strategy, including column-wise missing values before and after operations.
  """

  # Check for missing values
  if not df.isnull().values.any():
    print("No missing values detected in the DataFrame.")
    return df

  while True:
    # Print initial missing value information (column-wise)
    missing_counts = df.isnull().sum()
    print("\nMissing values before operation:")
    print(missing_counts)

    # Print current data set shape
    print(f"\nCurrent data set shape: {df.shape}")

    # Get user input for continuous operations
    print("For imputation method select one of these ['mean', 'median', 'most_frequent', 'constant']")
    user_input = input("How do you want to handle missing values? (imputation method or 'drop by index (idx/i)' or 'quit (quit/q)'): ").lower()

    if user_input in ['quit', 'q']:
      break
    elif user_input in ['i', 'idx']:
      # Ask for column indexes (comma-separated) to drop rows with missing values
      column_choice = input("Enter column indexes (comma-separated) to drop rows with missing values: ")
      if column_choice:
        try:
          # Convert user input to integer indexes
          selected_indexes = [int(idx) for idx in column_choice.split(',')]
          # Validate indexes within valid range
          if all(0 <= idx < len(df.columns) for idx in selected_indexes):
            valid_columns = df.columns[selected_indexes]
            # Drop rows with missing values in specified columns (by index)
            df = df.dropna(subset=valid_columns)
            print(df.shape)
          else:
            print("Invalid column indexes. Please enter indexes within the range 0 to", len(df.columns) - 1)
        except ValueError:
          print("Invalid input. Please enter comma-separated integers for column indexes.")
      else:
        print("No index specified. Please enter column indexes (comma-separated) to drop rows with missing values.")
    elif user_input in ['mean', 'median', 'most_frequent', 'constant']:
      strategy = user_input
      # Impute missing values using the chosen strategy
      imputer = SimpleImputer(strategy=strategy)
      imputer.fit(df)
      df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
      df = df_imputed  # Update dataframe with imputed values
      print(f"Missing values imputed using '{strategy}' strategy.")
    else:
      print("Invalid input. Please enter 'drop by index' for dropping rows, a valid imputation strategy (mean, median, most_frequent, constant), or 'quit' to exit.")

    # Print missing value information (column-wise) after operation
    missing_counts = df.isnull().sum()
    print("\nMissing values after operation:")
    print(missing_counts)

    # Print current data set shape after operation
    print(f"\nCurrent data set shape: {df.shape}")

  # Print final data set shape
  print(f"\nFinal data set shape: {df.shape}")
  return df

def handle_duplicates(df, subset=None, keep='first'):
  """
  Removes duplicate rows from a DataFrame.

  Args:
      df (pandas.DataFrame): The DataFrame containing potential duplicates.
      subset (list, optional): A list of column names to consider for duplicate checking.
          If None (default), all columns are considered.
      keep (str, optional): How to handle the first occurrence of duplicates.
          Defaults to 'first' (keep the first occurrence, remove subsequent ones).
          Other options include 'last' (keep the last occurrence) or False (remove all duplicates).

  Returns:
      pandas.DataFrame: The DataFrame with duplicates removed based on user selection.
  """

  # Check for duplicates
  if df.duplicated().sum() == 0:
    print("No duplicates found in the DataFrame.")
    return df

  # Get initial number of duplicates
  initial_duplicates = df.duplicated().sum()
  print(f"Found {initial_duplicates} duplicate rows in the DataFrame.")

  # Ask user if they want to review duplicates
  user_input = input("Do you want to review duplicates before removal? (y/n): ").lower()
  if user_input == 'y':
    print("\nHere are some options for handling duplicates:")
    print("- 'first' (default): Keep the first occurrence of duplicates.")
    print("- 'last': Keep the last occurrence of duplicates.")
    print("- 'specific': Select specific rows to keep (manual selection).")
    print("- 'remove all': Remove all duplicates.")

    while True:
      user_choice = input("Enter your choice (or 'q' to quit): ").lower()
      if user_choice == 'q':
        break
      elif user_choice in ['first', 'last', 'remove all']:
        keep = user_choice
        break
      elif user_choice == 'specific':
        # Implement logic for user selection of specific rows to keep (e.g., row indices)
        print("Functionality for selecting specific rows to keep is not yet implemented.")
        # Replace with code to allow user selection and update 'keep' variable
        continue
      else:
        print("Invalid input. Please choose a valid option.")

  # Remove duplicates based on user choice or default
  df_deduplicated = df.drop_duplicates(subset=subset, keep=keep)

  # Print results
  num_removed = initial_duplicates - df_deduplicated.duplicated().sum()
  print(f"Removed {num_removed} duplicate rows using '{keep}' strategy.")
  return df_deduplicated

def handle_outliers(df, column, method='iqr', threshold=3.5, action='remove', replacement_value=None):
    """
    Handles outliers in a DataFrame column using a specified method.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column with outliers.
        column (str): The name of the column containing potential outliers.
        method (str, optional): The method used to identify outliers. Defaults to 'iqr' (Interquartile Range).
            Other options include 'zscore' or a custom function that takes the DataFrame and column as arguments and returns a boolean Series indicating outliers.
        threshold (float, optional): The threshold for outlier detection based on the chosen method.
            Defaults to 3.5 for IQR. Adjust accordingly for other methods.
        action (str, optional): The action to take for handling outliers. Defaults to 'remove'.
            Other options include 'cap' (capping outliers), 'replace' (replacing outliers with a specified value), or 'skip' (do nothing).
        replacement_value (float, optional): The value to use for replacing outliers if 'action' is 'replace'. Defaults to None.

    Returns:
        pandas.DataFrame: The DataFrame with outliers handled according to the specified method and action.
    """

    # Check if the column is numerical before using IQR or zscore
    if method in ['iqr', 'zscore'] and not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Warning: '{method}' outlier detection is not suitable for non-numerical columns. Consider alternative methods or domain knowledge for '{column}'.")
        return df

    # Identify outliers based on the chosen method
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = np.abs(z_scores) > threshold
    elif callable(method):
        outliers = method(df, column)
    else:
        raise ValueError("Invalid outlier detection method. Choose 'iqr', 'zscore', or a custom function.")

    num_outliers = outliers.sum()
    print(f"Identified {num_outliers} potential outliers in '{column}' using '{method}' with threshold {threshold}.")

    # Handle outliers based on the specified action
    if action == 'remove':
        df_filtered = df[~outliers]
        print(f"Removed {num_outliers} outliers.")
        return df_filtered
    elif action == 'cap':
        capped_df = df.clip(lower=lower_bound, upper=upper_bound, inplace=False)
        print(f"Outliers capped between {lower_bound:.2f} and {upper_bound:.2f}.")
        return capped_df
    elif action == 'replace':
        if replacement_value is None:
            raise ValueError("Replacement value must be specified when action is 'replace'.")
        df.loc[outliers, column] = replacement_value
        print(f"Replaced {num_outliers} outliers with '{replacement_value}'.")
        return df
    elif action == 'skip':
        print("Skipping outlier handling.")
        return df
    else:
        raise ValueError("Invalid action. Choose 'remove', 'cap', 'replace', or 'skip'.")

def change_dtype(df):
    """
    Prompts the user to change data types of columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the columns.

    Returns:
        pandas.DataFrame: The DataFrame with potentially modified data types based on user choices.
    """

    # Print list of columns and data types
    print("Current column data types:")
    print(df.dtypes)

    # Ask user if they want to change data types
    user_input = input("Do you want to change data types of any columns? (y/n): ").lower()
    if user_input != 'y':
        return df  # Return original DataFrame if user chooses not to change

    while True:
        # Display available columns for user selection
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col} ({df.dtypes[i]})")  # Include current data type

        # Get user input for column selection
        while True:
            try:
                column_index = int(input("Enter the index of the column to change data type (or 'q' to quit): ")) - 1
                if 0 <= column_index < len(df.columns):
                    selected_column = df.columns[column_index]
                    break
                else:
                    print("Invalid column index. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

        if selected_column == 'q':
            break

        # Present conversion options based on current data type
        current_dtype = df[selected_column].dtype
        conversion_options = {
            'object': ['string', 'category', 'datetime64[ns]'],
            'int64': ['float64'],
            'float64': ['int64'],
            'datetime64[ns]': ['string']
        }
        if current_dtype not in conversion_options:
            print(f"Data type conversion for '{current_dtype}' is not currently supported.")
            continue

        print(f"\nCurrent data type of '{selected_column}' is: {current_dtype}")
        print("Available conversion options:")
        for option in conversion_options[current_dtype]:
            print(f"- {option}")

        # Get user input for new data type (string to date or string to numeric)
        while True:
            new_dtype = input("Enter the desired new data type (or 'q' to quit): ").lower()
            if new_dtype == 'q':
                break
            if new_dtype not in conversion_options[current_dtype]:
                print(f"Invalid data type choice. Please choose from available options.")
                continue

            # Handle string to date conversion
            if new_dtype == 'datetime64[ns]' and current_dtype == 'object':
                try:
                    # Prompt for date format if converting to datetime
                    date_format = input("Enter the date format (e.g., YYYY-MM-DD, %d/%m/%Y): ")
                    df[selected_column] = pd.to_datetime(df[selected_column], format=date_format)
                    print(f"Data type of column '{selected_column}' changed to '{new_dtype}'.")
                    break
                except pd.errors.ParserError:
                    print(f"Error parsing dates in '{selected_column}'. Please check your data and date format.")
                    continue

            # Handle string to numeric conversion (assuming other options are already supported by pandas)
            elif new_dtype in ['int64', 'float64'] and current_dtype == 'object':
                try:
                    # Attempt conversion and handle potential errors
                    df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce')
                    print(f"Data type of column '{selected_column}' changed to '{new_dtype}'. (Non-numeric values converted to NaN)")
                    break
                except ValueError:
                    print(f"Error converting '{selected_column}' to numeric type. Data might contain non-numeric values.")
                    confirmation = input("Are you sure you want to continue (y/n)? ").lower()
                    if confirmation != 'y':
                        continue
            else:
                df[selected_column] = df[selected_column].astype(new_dtype)
                print(f"Data type of column '{selected_column}' changed to '{new_dtype}'.")
                break
        else:
            print(f"Invalid data type choice. Please choose from available options.")
            continue

    return df

def select_features(df):
  """
  Selects a subset of columns from a DataFrame based on user input.

  Args:
      df (pandas.DataFrame): The DataFrame containing the columns.

  Returns:
      pandas.DataFrame: The DataFrame with the selected columns.
  """

  # Display available columns for user reference
  print("Available columns:")
  for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

  selected_indices = []
  while True:
    user_input = input("Enter the index of a column to select (or 'q' to finish): ")
    if user_input.lower() == 'q':
      break
    try:
      index = int(user_input) - 1  # Adjust for 0-based indexing
      if 0 <= index < len(df.columns):
        selected_indices.append(index)
      else:
        print("Invalid column index. Please try again.")
    except ValueError:
      print("Invalid input. Please enter a number or 'q' to quit.")

  if not selected_indices:
    print("No columns selected.")
    return df  # Return original DataFrame if no selection is made

  selected_columns = df.iloc[:, selected_indices]
  print("Selected columns:")
  for col in selected_columns.columns:
    print(col)

  return selected_columns

def handle_categorical_variables(df):
  """
  Converts categorical variables in a DataFrame to dummy variables using pd.get_dummies().

  Args:
      df (pandas.DataFrame): The DataFrame containing categorical variables.

  Returns:
      pandas.DataFrame: The DataFrame with categorical variables converted to dummy variables,
                        or the original DataFrame if errors occur. (Minimizes potential errors)

  Prints:
      A message indicating successful conversion or encountered errors.
  """

  # Identify potential categorical columns with additional checks
  categorical_cols = []
  for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):  # Ensure non-numeric
      try:
        # Check uniqueness and handle potential errors (e.g., mixed data types)
        nunique_values = df[col].nunique(dropna=False)
        if nunique_values < 10 and pd.api.types.is_string_dtype(df[col]):
          categorical_cols.append(col)
      except (KeyError, TypeError):  # Handle missing columns or incompatible data
        print(f"Warning: Potential error checking column '{col}'. Skipping.")

  if categorical_cols:
    # Create dummy variables if categorical columns are found
    try:
      df_with_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
      print("Converted categorical variables to dummy variables:")
      print(df_with_dummies.head())
      return df_with_dummies
    except ValueError as e:
      print(f"Error converting categorical variables to dummy variables: {e}")
      return df  # Return original DataFrame if error occurs

  else:
    print("No categorical variables found for conversion.")
    return df  # Return original DataFrame if no categorical variables

def save_datasets(dataframes, filenames):
  """
  Saves a list of DataFrames to CSV files with specified names.

  Args:
      dataframes (list): A list of pandas DataFrames to be saved.
      filenames (list): A list of strings containing the desired filenames for the CSV files.
  """

  if len(dataframes) != len(filenames):
      print("Error: Number of DataFrames and filenames must match.")
      return
  
  for i, (df, filename) in enumerate(zip(dataframes, filenames)):
      # Construct full path using current working directory
      full_path = os.path.join(os.getcwd(), filename )

      # Validate filename (checks for valid path)
      if not os.path.dirname(full_path):
          print(f"Error: Invalid filename '{filename}'. Please provide a valid path.")
          continue  # Skip saving this DataFrame

      try:
          df.to_csv(full_path, index=False)
          print(f"DataFrame saved to '{full_path}'.")
      except IOError as e:
          print(f"Error saving DataFrame to '{full_path}': File not found.")
      except PermissionError as e:
          print(f"Error saving DataFrame to '{full_path}': Permission denied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data cleaning and preprocessing script.')
    parser.add_argument('file_paths', metavar='file_paths', type=str, nargs='+',
                        help='File paths to data files separated by space')
    args = parser.parse_args()
    main(args.file_paths)
