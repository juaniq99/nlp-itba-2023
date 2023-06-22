import pandas as pd


# No tokenizer here because predict method does it for us
# Here we only need functions to retrieve comments from dataset

class MoviesUtils:
    dataset_file = "movies/dataset_labeled.csv"

    # Returns n comments from dataset (for prediction)
    @classmethod
    def get_lines(cls, count, seed=17):
        dataset_df = cls.get_dataset()
        # Shuffle
        dataset_df = dataset_df.sample(frac=1, random_state=seed)
        # Return comment and labels
        return dataset_df[:count]
    
    # Returns n comments with tagged labels
    @classmethod
    def get_tagged_lines(cls, count, seed=17):
        dataset_df = cls.get_dataset()
        # Drop rows with no label
        dataset_df = dataset_df.dropna(subset=['label_openness', 'label_conscientiousness', 'label_extraversion', 'label_agreeableness', 'label_neuroticism'])
        # Shuffle
        dataset_df = dataset_df.sample(frac=1, random_state=seed)
        # Return comment and labels
        return dataset_df[:count]
    
    
    @classmethod
    def get_dataset(cls):
        # Load csv into pandas
        dataset_df = pd.read_csv(cls.dataset_file)

        # Check if body column is empty
        if dataset_df['text'].isnull().values.any():
            dataset_df = dataset_df.dropna(subset=['text'])

        return dataset_df
    
