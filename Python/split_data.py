import numpy as np
from generator import signalLoader

def split_data(classes,k,i,the_names,path,spex,batch_size=1,class_on_char=0):
    np.random.shuffle(the_names)
    class_names = []
    class_names_val = []
    for c in classes:
        names = [idx for idx in the_names if idx[class_on_char].lower() == c.lower()]
        list_names = np.array_split(names,k)
        class_names.extend(np.hstack(np.delete(list_names, i, 0)).transpose())
        class_names_val.extend(list_names[i])
    np.random.shuffle(class_names)
    np.random.shuffle(class_names_val)
    genVal = signalLoader(class_names_val,path,spex,batch_size=batch_size,class_on_char=class_on_char)
    gen = signalLoader(class_names,path,spex,batch_size=batch_size,class_on_char=class_on_char)
    return (gen,genVal,len(class_names),len(class_names_val))
