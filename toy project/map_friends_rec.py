#!/usr/bin/env python3
#Map task for the Friend recommendation problem

import itertools
import sys

# input comes from STDIN (standard input)

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    line = line.split()
    key = int(line[0])
    if len(line)>1:
        friends = line[1]
        if friends!='':
            friends = line[1].split(",")
            friends = sorted(map(int,friends))
            for friend in friends:
                pair = tuple(sorted([key,friend]))
                pair = ','.join(map(str,pair))
                print(pair,"\t",1)
            for pair in itertools.combinations(friends,2):
                pair = ','.join(map(str,pair))
                print(pair,"\t",0)
