import pandas as pd
from bs4 import BeautifulSoup
import re


imdb_data = pd.read_csv('IMDB Dataset.csv')

# removing the html lines
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(denoise_text)

# drop the first column of serial numbers
#imdb_data = imdb_data.drop(imdb_data.columns[[0]], axis=1)

# Print
print("processed the data and now writing back to the file")

#write back to another csv file
imdb_data.to_csv('processed_imdb_data', index=False)
