from transformers_code import train, predict
from transformers_code.utils.pandoraUtils import PandoraUtils
from transformers_code.utils.moviesUtils import MoviesUtils

if __name__ == "__main__":
    # Train
    #train.run_one_trait_pandora("neuroticism")
    #train.run_kcross_one_trait_pandora("openness", 5)
    
    # Predict
    data = MoviesUtils.get_lines(10000000)      # There are 5139 (grouped) lines in the dataset
    predict.predict("saved_models/final_neuroticism",'text', data, data["label_neuroticism"].to_list())

