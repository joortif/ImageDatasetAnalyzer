import os 
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasetanalyzerlib')))

from imagedatasetanalyzer.src.embeddings.tensorflowembedding import TensorflowEmbedding

def _clean_model_name(model_name) -> str:
        """
        Cleans and returns a simplified version of the model name by removing certain version identifiers, 
        size descriptors, and digits that are not part of the model's name.

        Returns:
        str: The cleaned and simplified model name, with sizes and unnecessary numbers removed.
        """
        model_name = model_name.lower()
        versions = ["v2", "v3"]
        sizes = ["tiny", "small", "base", "xlarge", "large" , "mobile"]

        for size in sizes:
            if model_name.endswith(size):
                model_name = model_name.replace(size, '')

        name_without_digits = re.sub(r'(?<!v)\d+', '', model_name)

        for version in versions:
            if version in name_without_digits:
                name_without_digits = name_without_digits.replace(version, f"_{version}")

        return name_without_digits.strip()

def test_model_name_cleaner():
    model_names = [
        "ConvNeXtBase", "ConvNeXtLarge", "ConvNeXtSmall", "ConvNeXtTiny", "ConvNeXtXLarge",
        "DenseNet121", "DenseNet152", "DenseNet201", "NASNetLarge", "NASNetMobile",
        "ResNet101", "ResNet50", "ResNet152", "ResNet101V2", "ResNet152V2", "ResNet50V2", "MobileNetV3"
    ]

    for model_name in model_names:
        cleaned_name = _clean_model_name(model_name)
        print(f"Original: {model_name} -> Cleaned: {cleaned_name}")

if __name__ == "__main__":
    test_model_name_cleaner()