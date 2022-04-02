from utils import text_process


def simple_sentences_rotation(dataset_path, dataset_dest):
    """Create a new dataset based on another one with rotated sentences"""
    rotated_sentences = []

    # get all sentences of the dataset in a list
    original_sentences = text_process.get_tokenlized(dataset_path)

    for sentence in original_sentences:
        for i in range(5):
            if sentence[-1] != '.':
                rotated_sentence = sentence[i + 1:] + sentence[:i + 1]
                rotated_sentences.append(rotated_sentence)
            else:
                rotated_sentence = sentence[i + 1:-1] + sentence[:i + 1] + list('.')
                rotated_sentences.append(rotated_sentence)

    # Write original and new sentences into a file
    text_process.write_tokens(dataset_dest, original_sentences)
    text_process.write_tokens(dataset_dest, rotated_sentences)


# The idea will be to rename the new file to image_coco so that no code has to be modified
simple_sentences_rotation("image_coco.txt", "image_coco_with_rotations.txt")
