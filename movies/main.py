import pandas as pd
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re

outfile = "dataset_labeled.csv"


def clean_text(body):
    # Remove urls, subreddit links and [] which are hyperlinks
    #body = re.sub(r'\(?http\S+', '', body)
    #body = re.sub(r'/?r/\S+', '', body).replace('[', '').replace(']', '').strip()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(body)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    body = ' '.join(filtered_sentence)
    # One last strip to remove any extra spaces in case comment was only stopwords, to reduce to empty string
    body = body.strip()
    return body


def group_comments(df):
    # Result dataframe
    new_df = pd.DataFrame(columns=['speaker_id', 'text', 'og_text'])
    # Group by author
    grouped = df.groupby('speaker_id')

    # Join all comments of same author
    for name, group in grouped:
        new_body = ""
        new_og_body = ""
        for index, row in group.iterrows():
            new_body += row['text'] + " "
            new_og_body += row['og_text'] + " "
            # Build at least 300 word comments
            if len(new_body.split()) >= 300:  # Push to df
                # Remove last space
                new_body = new_body[:-1]
                new_og_body = new_og_body[:-1]
                new_row = {'speaker_id': name, 'text': new_body, 'og_text': new_og_body}
                new_row_df = pd.DataFrame(new_row, index=[0])
                new_df = pd.concat([new_df, new_row_df], ignore_index = True)
                # Reset
                new_body = ""
                new_og_body = ""

    return new_df


if __name__ == "__main__":

    # Read lines file into dataframe
    lines_file = "D:\\Downloads\\nlp-itba-2023\\movies\\lines.csv"
    lines_df = pd.read_csv(lines_file)
    
    # Read speakers file into dataframe
    speakers_file = "D:\\Downloads\\nlp-itba-2023\\movies\\speakers_tagged.csv"
    speakers_df = pd.read_csv(speakers_file)

    # Drop if body is empty
    lines_df = lines_df[lines_df['text'] != '']

    # Drop if body is None
    lines_df = lines_df[lines_df['text'].notna()]

    # Save original text
    lines_df.rename(columns = {'text':'og_text'}, inplace = True)

    # Clean text
    lines_df['text'] = lines_df['og_text'].apply(clean_text)

    # Group comments on longer ones, so as to not work with short sentences
    lines_df = group_comments(lines_df)

    # Drop if body is empty
    lines_df = lines_df[lines_df['text'] != '']

    # Drop if body is None
    lines_df = lines_df[lines_df['text'].notna()]

    # Merge comments and users on author
    df_merged = pd.merge(lines_df, speakers_df, on='speaker_id', how='inner')

    # Print 30 first rows of merged dataframe
    #print("\nDf head: \n")
    #print(df_merged.head(30))

    # Write to outfile. Write comment and label
    df_merged.to_csv(outfile, columns=['text', 'og_text', 'speaker_id', 'name', 'gender', 'movie_id', 'movie_name', 'label_agreeableness', 'label_openness', 'label_conscientiousness', 'label_extraversion', 'label_neuroticism'], index=False, header=True, encoding='utf-8')

