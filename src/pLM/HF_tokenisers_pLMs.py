from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def ankh_base_tokeniser():
    tokeniser = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
    return tokeniser


def ankh_base_model_for_eval():
    model = AutoModelForSeq2SeqLM.from_pretrained("ElnaggarLab/ankh-base")
    # freeze weights for inference mode:
    return model.eval()

