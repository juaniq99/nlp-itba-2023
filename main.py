from transformers_code import train, predict
from transformers_code.utils.pandoraUtils import PandoraUtils

if __name__ == "__main__":
    #run.run_32_labels_pandora()
    train.run_one_trait_pandora("neuroticism")
    #train.run_kcross_one_trait_pandora("openness", 5)
    #predict.predict("saved_models/20230511_182453_openness_pandora", ["This is a comment", "This is another"])

    #data = PandoraUtils.get_dataset()
    # Print size
    #print(f"Dataset size: {len(data)}")

    #data = PandoraUtils.get_unseen_comments(1000000, 92800, "openness", balanced=False)
    # Print size
    #print(f"Unseen dataset size: {len(data)}")


    #data = PandoraUtils.get_test_comments(10000, "openness")
    #predict.predict("saved_models/tuneado_openness",'tweet', data, data["label_openness"].to_list())