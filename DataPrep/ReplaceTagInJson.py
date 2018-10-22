# ==============================================================================
# Tool set to convert annotation to VoTT json file
#
# Shu Peng
# ==============================================================================
import os
import sys
import json


def main(argv):
    if len(argv) < 3:
        print("Syntax: python {} <json_file> <old_string> <new_string>".format(os.path.basename(argv[0])))
    else:
        j_file = argv[1]
        old_string = argv[2]
        new_string = argv[3]
        if not os.path.exists(j_file):
            print("File not found : {}".format(j_file))
        else:
            print("Open {}".format(j_file))
            print("Replacing VoTT tag '{}' with '{}'.".format(old_string, new_string))
            with open(j_file, 'r') as jf:
                obj = json.load(jf)
            tag_string = obj['inputTags']
            tag_string = tag_string.replace(old_string, new_string)
            obj['inputTags'] = tag_string
            counter = 0
            for frame in obj['frames']:
                for b in range(len(obj['frames'][frame])):
                    for t in range(len(obj['frames'][frame][b]['tags'])):
                        if obj['frames'][frame][b]['tags'][t] == old_string:
                            obj['frames'][frame][b]['tags'][t] = new_string
                            counter += 1
            with open(j_file, 'w') as jf:
                json.dump(obj, jf)
            print("{:d} tag(s) were replaced.".format(counter))


if __name__ == '__main__':
    main(sys.argv)
