from transformers_code import train, predict
from transformers_code.utils.pandoraUtils import PandoraUtils
from transformers_code.utils.moviesUtils import MoviesUtils

if __name__ == "__main__":
    #run.run_32_labels_pandora()
    #train.run_one_trait_pandora("neuroticism")
    #train.run_kcross_one_trait_pandora("openness", 5)
    #predict.predict("saved_models/20230511_182453_openness_pandora", ["This is a comment", "This is another"])

    #data = PandoraUtils.get_dataset()
    # Print size
    #print(f"Dataset size: {len(data)}")

    #data = PandoraUtils.get_unseen_comments(1000000, 92800, "openness", balanced=False)
    # Print size
    #print(f"Unseen dataset size: {len(data)}")


    #data = PandoraUtils.get_test_comments(10, "openness")
    #predict.predict("saved_models/final_openness",'body', data, data["label_openness"].to_list())

    data = MoviesUtils.get_lines(10000000)      # There are 5139 (grouped) lines in the dataset
    #print(data)
    #data = MoviesUtils.get_tagged_lines(1000)
    #print(data)
    # Print column "name" - Shows the name of the character
    #print(data["name"])
    predict.predict("saved_models/final_agreeableness",'text', data, data["label_agreeableness"].to_list())

