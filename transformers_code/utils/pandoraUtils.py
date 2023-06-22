from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold


class PandoraUtils:
    #tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    dataset_file = "pandora/dataset_labeled.csv"


    @classmethod
    def tokenize_one_trait(cls,batch, trait_name):
        tokenized_batch = cls.tokenizer(batch["body"], padding="max_length", truncation=True)
        # Map labels from string to int
        tokenized_batch["labels"] = [int(label) for label in batch["label_" + trait_name]]
        return tokenized_batch


    @classmethod
    def get_tokenized_dataset(cls,dataset_size,test_size):
        
        traits=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

        #Generate different datasets based on b5 factor
        binary_datasets = {}

        for trait in traits:
            binary_datasets[trait] = cls.get_tokenized_dataset_one_trait(trait,dataset_size,test_size)

        return binary_datasets
    

    @classmethod
    def get_dataset(cls):
        # Load csv into pandas
        dataset_df = pd.read_csv(cls.dataset_file)

        # Check if body column is empty
        if dataset_df['body'].isnull().values.any():
            dataset_df = dataset_df.dropna(subset=['body'])

        return dataset_df


    @classmethod
    def get_tokenized_dataset_one_trait(cls,trait_name,dataset_size,test_size):
        dataset_df = cls.get_dataset()

        # Balance dataset, keeping desired size
        g = dataset_df.groupby('label_' + trait_name)

        # Get number of unique labels
        number_of_labels = dataset_df['label_' + trait_name].nunique()
        
        desired_label_size = g.size().min() # Either size of smallest group or dataset_size/number_of_labels
        if g.size().min() > int(dataset_size/number_of_labels):
            desired_label_size = int(dataset_size/number_of_labels)

        random_state = 17
        dataset_df = pd.DataFrame(g.apply(lambda x: x.sample(desired_label_size, random_state=random_state).reset_index(drop=True)))

        # Load into dataset object
        dataset = Dataset.from_pandas(dataset_df)

        # Print size
        print(f"Dataset size: {len(dataset)}")

        # Print label count
        print(f"Label count: {pd.Series(dataset['label_' + trait_name]).value_counts()}")

        # Split into train and test
        dataset = dataset.train_test_split(test_size=test_size, seed=17)

        # Tokenize
        tokenize_func = lambda batch: cls.tokenize_one_trait(batch, trait_name)
        tokenized_dataset = dataset.map(tokenize_func, batched=True)

        return tokenized_dataset
    

    # Returns a list of k tokenized datasets, each one with a different test set
    # Maybe this occupies too much memory, switched to one partition per project run
    @classmethod
    def get_kcross_tokenized_dataset_one_trait(cls,trait_name,dataset_size,k):
        dataset_df = cls.get_dataset()

        # Balance dataset, keeping desired size
        g = dataset_df.groupby('label_' + trait_name)

        # Get number of unique labels
        number_of_labels = dataset_df['label_' + trait_name].nunique()
        
        desired_label_size = g.size().min() # Either size of smallest group or dataset_size/number_of_labels
        if g.size().min() > int(dataset_size/number_of_labels):
            desired_label_size = int(dataset_size/number_of_labels)

        random_state = 17
        dataset_df = pd.DataFrame(g.apply(lambda x: x.sample(desired_label_size, random_state=random_state).reset_index(drop=True)))

        # Load into dataset object
        dataset = Dataset.from_pandas(dataset_df)

        # Print size
        print(f"Dataset size: {len(dataset)}")

        # Print label count
        print(f"Label count: {pd.Series(dataset['label_' + trait_name]).value_counts()}")

        # Split into train and test k times
        folds = StratifiedKFold(n_splits=k)     # Stratified to keep distribution of labels (balanced partitions)
        splits = folds.split(dataset_df, dataset_df['label_' + trait_name])
        tokenized_datasets = []
        tokenize_func = lambda batch: cls.tokenize_one_trait(batch, trait_name)
        for train_idxs, test_idxs in splits:
            dataset = Dataset.from_pandas(dataset_df)
            fold_dataset = DatasetDict({
                "train":dataset.select(train_idxs),
                "test":dataset.select(test_idxs)
            })
            # Tokenize
            tokenized_dataset = fold_dataset.map(tokenize_func, batched=True)
            tokenized_datasets.append(tokenized_dataset)

        return tokenized_datasets
    

    # Returns one partition of a k-cross tokenized dataset
    @classmethod
    def get_one_kcross_tokenized_dataset_one_trait(cls,trait_name,dataset_size,k,curr_k):
        dataset_df = cls.get_dataset()

        # Balance dataset, keeping desired size
        g = dataset_df.groupby('label_' + trait_name)

        # Get number of unique labels
        number_of_labels = dataset_df['label_' + trait_name].nunique()
        
        desired_label_size = g.size().min() # Either size of smallest group or dataset_size/number_of_labels
        if g.size().min() > int(dataset_size/number_of_labels):
            desired_label_size = int(dataset_size/number_of_labels)

        random_state = 17
        dataset_df = pd.DataFrame(g.apply(lambda x: x.sample(desired_label_size, random_state=random_state).reset_index(drop=True)))

        # Split into train and test k times
        folds = StratifiedKFold(n_splits=k)
        splits = folds.split(dataset_df, dataset_df['label_' + trait_name])
        tokenize_func = lambda batch: cls.tokenize_one_trait(batch, trait_name)
        
        # Get the curr_k partition
        for i, (train_idxs, test_idxs) in enumerate(splits):
            if i == curr_k:
                dataset = Dataset.from_pandas(dataset_df)
                fold_dataset = DatasetDict({
                    "train":dataset.select(train_idxs),
                    "test":dataset.select(test_idxs)
                })
                # Tokenize
                tokenized_dataset = fold_dataset.map(tokenize_func, batched=True)
                return tokenized_dataset

        # Should always match a curr_k
        raise Exception("curr_k not found")


    # Returns n comments from dataset (for prediction)
    @classmethod
    def get_comments(cls, count, seed=17):
        dataset_df = cls.get_dataset()
        # Shuffle
        dataset_df = dataset_df.sample(frac=1, random_state=seed)
        # Return comment and labels
        return dataset_df[:count]
    

    # Returns n comments from dataset that model didn't train nor tested with (for prediction)
    # Dataset size, trait name and random_state (not seed param) must be the same as used in training
    @classmethod
    def get_unseen_comments(cls, count, train_dataset_size, trait_name, balanced=False, seed=17):
        dataset_df = cls.get_dataset()

        # Get number of unique labels
        number_of_labels = dataset_df['label_' + trait_name].nunique()

        # Balance dataset, keeping desired size
        g = dataset_df.groupby('label_' + trait_name)
        
        desired_label_size = g.size().min() # Either size of smallest group or dataset_size/number_of_labels
        if g.size().min() > int(train_dataset_size/number_of_labels):
            desired_label_size = int(train_dataset_size/number_of_labels)

        random_state = 17
        training_data = pd.DataFrame(g.apply(lambda x: x.sample(desired_label_size, random_state=random_state)))
        #unseen_data = dataset_df[~dataset_df.isin(training_data)]
        unseen_data = pd.concat([dataset_df, training_data]).drop_duplicates(keep=False)

        # Balance the ones we will return, if required
        if balanced:
            # Balance dataset, keeping desired size
            g = unseen_data.groupby('label_' + trait_name)
            desired_label_size = g.size().min()
            if g.size().min() > int(count/number_of_labels):
                desired_label_size = int(count/number_of_labels)
            unseen_data = pd.DataFrame(g.apply(lambda x: x.sample(desired_label_size, random_state=seed)))
        else:
            # Shuffle
            unseen_data = unseen_data.sample(frac=1, random_state=seed)
        
        # Return comment and labels
        return unseen_data[:count]
    

    # Returns the comments used for testing when the model trained
    @classmethod
    def get_test_comments(cls, train_dataset_size, trait_name, test_size=0.2):
        dataset_df = cls.get_dataset()

        # Balance dataset, keeping desired size
        g = dataset_df.groupby('label_' + trait_name)

        # Get number of unique labels
        number_of_labels = dataset_df['label_' + trait_name].nunique()
        
        desired_label_size = g.size().min() # Either size of smallest group or dataset_size/number_of_labels
        if g.size().min() > int(train_dataset_size/number_of_labels):
            desired_label_size = int(train_dataset_size/number_of_labels)

        random_state = 17
        dataset_df = pd.DataFrame(g.apply(lambda x: x.sample(desired_label_size, random_state=random_state).reset_index(drop=True)))

        # Load into dataset object
        dataset = Dataset.from_pandas(dataset_df)

        # Split into train and test
        dataset = dataset.train_test_split(test_size=test_size, seed=17)

        # Keep test data
        dataset = dataset['test']

        # Convert into pandas
        df = dataset.to_pandas()
        df.drop(columns=['__index_level_0__', '__index_level_1__'], inplace=True)

        return df