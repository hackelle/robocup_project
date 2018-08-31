#!/usr/bin/env python3

# To sort by dataset, run
# ./whatsdone.py pics data.json | awk '{split($2,a,"_"); print a[1]}' | uniq -c

import json
import os
import sys

if len(sys.argv) < 3:
    print('Usage: {} PICS_DIR DATA_OUTPUTS...'.format(sys.argv[0]))
    sys.exit(64)

pics = map(lambda f: os.path.join(sys.argv[1], f), os.listdir(sys.argv[1]))

for path in sys.argv[2:]:
    pics_done = {}
    with open(path) as fh:
        new_data = json.load(fh)
        for pic in new_data:
            if pic in pics_done:
                where = pics_done[pic]
                print('WARNING: {} is in {} and {}!'.format(pic, where, path))
            else:
                pics_done[pic] = path

for pic in sorted(pics):
    if pic not in pics_done:
        print('MISSING: {}'.format(pic))
