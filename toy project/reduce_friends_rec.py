#!/usr/bin/env python3

#Reduce task for the Friend recommendation problem

import itertools
import sys

n_rec = 10

def addPair(users,k1,k2,friendFlag):
    if friendFlag==1:
        friendFlag = True
    else:
        friendFlag = False

    if k1 not in users:
        users[k1] = {}
        users[k1][k2] = [1,False]
    else:
        if k2 in users[k1]:
            users[k1][k2][0] += 1            
        else:
            users[k1][k2] = [1,False]
    if friendFlag==True:
        users[k1][k2][0] -= 1
        users[k1][k2][1] = True



# input comes from STDIN (standard input)
users = {}

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    line = line.split("\t")
    key = tuple(map(int,line[0].strip().split(",")))
    friendFlag = int(line[1])
    k1,k2 = key
    addPair(users,k1,k2,friendFlag)
    addPair(users,k2,k1,friendFlag)

for k1 in users.keys():
    recommendations = []
    for k2 in users[k1].keys():
        n,flag = users[k1][k2]
        if flag==False:
            recommendations.append((k2,n))
    recommendations = sorted(recommendations,key=lambda x: x[0])
    recommendations = sorted(recommendations,key=lambda x: x[1],reverse=True)
    if len(recommendations)>0:
        recommendations = list(map(str,list(zip(*recommendations))[0]))
        print(k1,"\t",','.join(recommendations[:n_rec]))