import os
dirpath = './ppl/results'
i = 1
for filename in os.listdir(dirpath):
    new_filename = str(i) + '.png'
    old_file = os.path.join(dirpath,filename)
    new_file = os.path.join(dirpath,new_filename)
    os.rename(old_file,new_file)
    i += 1