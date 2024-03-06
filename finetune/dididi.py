import pathlib

path = str(pathlib.Path(__file__).parent.absolute() / 'output')
print(path, type(path))