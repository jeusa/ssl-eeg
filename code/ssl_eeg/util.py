"""Useful tools."""

import os


def list_files(directory, suffix="", recursive=True, sort_by="name"):

    if not directory.endswith(os.sep):
        directory = directory + os.sep

    files = []
    dir_files = os.listdir(directory)

    rec = recursive
    if type(recursive) is int:
        rec = recursive-1

    for f in dir_files:

        if f.lower().endswith(suffix):
            files.append(directory + f)

        elif ((recursive == True) | (recursive >= 1)) & os.path.isdir(directory + f):
            sub_dir_files = list_files(directory + f, suffix, rec)
            files = files + sub_dir_files
    
    sort_key = None
    if sort_by == "name":
        sort_key = str.lower
    elif sort_by == "date":
        sort_key = os.path.getmtime
    
    if sort_by:
        files = sorted(files, key=sort_key)

    return files


def flatten(t):
    return [item for sublist in t for item in sublist]
