from utils import text_process


def simple_sentences_rotation(dataset_path):
    """Create a new dataset based on another one with rotated sentences"""
    rotated_sentences = []

    # get all sentences of the dataset in a list
    original_sentences = text_process.get_tokenlized(dataset_path)

    for sentence in original_sentences:
        for i in range(5):
            rotated_sentence = sentence[i + 1:] + sentence[:i + 1]
            rotated_sentences.append(rotated_sentence)

    # Write original and new sentences into a file
    text_process.write_tokens(..., original_sentences)
    text_process.write_tokens(..., rotated_sentences)
