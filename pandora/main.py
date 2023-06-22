import pandas as pd
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re

outfile = "dataset_labeled.csv"

# Add label to every row in dataframe
def add_labels(author_df):
    author_df[['label_openness', 'label_conscientiousness', 'label_extraversion', 'label_agreeableness', 
               'label_neuroticism']] = pd.DataFrame([[0, 0, 0, 0, 0]], index=author_df.index)
    for index, row in author_df.iterrows():
        if row['openness'] >= 50:
            author_df.loc[index, 'label_openness'] = 1
        if row['conscientiousness'] >= 50:
            author_df.loc[index, 'label_conscientiousness'] = 1
        if row['extraversion'] >= 50:
            author_df.loc[index, 'label_extraversion'] = 1
        if row['agreeableness'] >= 50:
            author_df.loc[index, 'label_agreeableness'] = 1
        if row['neuroticism'] >= 50:
            author_df.loc[index, 'label_neuroticism'] = 1


def clean_text(body):
    # Remove urls, subreddit links and [] which are hyperlinks
    body = re.sub(r'\(?http\S+', '', body)
    body = re.sub(r'/?r/\S+', '', body).replace('[', '').replace(']', '').strip()
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
    new_df = pd.DataFrame(columns=['author', 'body', 'og_body'])
    # Group by author
    grouped = df.groupby('author')

    # Join all comments of same author
    for name, group in grouped:
        new_body = ""
        new_og_body = ""
        for index, row in group.iterrows():
            new_body += row['body'] + " "
            new_og_body += row['og_body'] + " "
            # Build at least 300 word comments
            if len(new_body.split()) >= 300:  # Push to df
                # Remove last space
                new_body = new_body[:-1]
                new_og_body = new_og_body[:-1]
                new_row = {'author': name, 'body': new_body, 'og_body': new_og_body}
                new_row_df = pd.DataFrame(new_row, index=[0])
                new_df = pd.concat([new_df, new_row_df], ignore_index = True)
                # Reset
                new_body = ""
                new_og_body = ""

    return new_df


if __name__ == "__main__":
    
    file = open('config.json')
    json_values = json.load(file)
    file.close()

    users_file = json_values.get("users_file")
    big5_comments = json_values.get("big5_comments")   

    # Read users csv using pandas
    df_authors = pd.read_csv(users_file)
    filtered_authors = df_authors.loc[:, ['author','openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']]
    filtered_authors = filtered_authors.dropna()
    # Add label using big5 scores
    add_labels(filtered_authors)
    # Drop big5 scores - don't drop, better for analysis
    #filtered_authors = filtered_authors.drop(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'], axis=1)

    # Read comments csv using pandas, read only first 1000 lines
    df_comments = pd.read_csv(big5_comments, on_bad_lines='skip')
    df_comments = df_comments.loc[:, ['author', 'body', 'lang']]

    # Drop if lang is not en
    df_comments = df_comments[df_comments['lang'] == 'en']

    # Drop lang column
    df_comments = df_comments.drop(['lang'], axis=1)

    # Save original text
    df_comments.rename(columns = {'body':'og_body'}, inplace = True)

    # Clean text
    df_comments['body'] = df_comments['og_body'].apply(clean_text)

    # Group comments on longer ones, so as to not work with short sentences
    df_comments = group_comments(df_comments)

    # Merge comments and users on author
    df_merged = pd.merge(df_comments, filtered_authors, on='author', how='inner')

    # Drop if body is empty
    df_comments = df_comments[df_comments['body'] != '']

    # Drop if body is None
    df_comments = df_comments[df_comments['body'].notna()]

    # Print 30 first rows of merged dataframe
    #print("\nDf head: \n")
    #print(df_merged.head(30))

    # Write to outfile. Write comment and label
    df_merged.to_csv(outfile, columns=['body', 'og_body', 'label_openness', 'label_conscientiousness', 'label_extraversion', 'label_agreeableness', 'label_neuroticism', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'author'], index=False, header=True, encoding='utf-8')

