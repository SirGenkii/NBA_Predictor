#get latest file from CLEANED_DATA_DIR. All files end with current date with following format: '%Y-%m-%d_%H-%M-%S'

# match_files = os.listdir(CLEANED_DATA_DIR)

# #get last generated file
# last_file = sorted(match_files)[-1]

# ALL_ATP_MATCHES = os.path.join(CLEANED_DATA_DIR, last_file)
# print("Latest match file found : ", os.path.join(CLEANED_DATA_DIR, last_file))


#function to save a dataframe to a csv file in dir parameter, with pr