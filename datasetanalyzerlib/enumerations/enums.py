from enum import Enum

from datasetanalyzerlib.exceptions.exceptions import ExtensionNotFoundException

class Extensions(Enum):
    JSON = '.json'
    TXT = '.txt'
    PNG = '.png'
    JPG = '.jpg'
    XML = '.xml'

    def extensionToEnum(ext):
        for extension in Extensions:
            if ext.lower() == extension.value.lower():  
                return extension
        raise ExtensionNotFoundException(ext)
    
    def enumToExtension(enum):
        return enum.value
