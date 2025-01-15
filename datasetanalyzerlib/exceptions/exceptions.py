class ExtensionNotFoundException(Exception):
    def __init__(self, ext):
        self.ext = ext
        self.message = f"Extension '{self.ext}' not found in the supported extensions."
        super().__init__(self.message)