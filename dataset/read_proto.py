"""
test for read coco_names.prototxt

"""
from google.protobuf import text_format
from alfred.protos.labelmap_pb2 import LabelMap

map_dict = {}

with open('coco.prototxt', 'r') as f:
    lm = LabelMap()
    lm = text_format.Merge(str(f.read()), lm)
    names_list = [i.display_name for i in lm.item]
    
    for i in lm.item:
        map_dict[i.id] = names_list.index(i.display_name)
    print(map_dict)