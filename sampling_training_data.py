import os
import json
import random

def main():
    os.chdir("./dataset/trp/data/")

    with open('train.json') as json_file:
        data = json.load(json_file)

    number = {
        "TrAP"      : 0,
        "TrCP"      : 0,
        "TrIP"      : 0,
        "TrNAP"     : 0,
        "TrWP"      : 0,
        "NoRE_TrP"  : 0
    }

    for row in data:
        number[row["relation"]] += 1
    print(number)

    with open('train_bak.json', 'w') as outfile:
        json.dump(data, outfile)

    new_data = []
    count=0
    for row in data:
        if(row["relation"] == "TrAP"):
            new_data.append(row)
        if(row["relation"] == "TrCP"):
            for _ in range(4):
                new_data.append(row)
        if(row["relation"] == "TrIP"):
            for _ in range(18):
                new_data.append(row)
        if(row["relation"] == "TrNAP"):
            for _ in range(14):
                new_data.append(row)
        if(row["relation"] == "TrWP"):
            for _ in range(43):
                new_data.append(row)
        if(row["relation"] == "NoRE_TrP"):
            count+=1
            if count%5==0:
                continue
            new_data.append(row)

    number = {
        "TrAP"      : 0,
        "TrCP"      : 0,
        "TrIP"      : 0,
        "TrNAP"     : 0,
        "TrWP"      : 0,
        "NoRE_TrP"  : 0
    }

    for row in new_data:
        number[row["relation"]] += 1

    random.shuffle(new_data)

    with open('train.json', 'w') as outfile:
        json.dump(new_data, outfile)

    print(number)
    print(sum(number.values()) / 6)

    os.chdir("../tep/data/")

    with open('train.json') as json_file:
        data = json.load(json_file)

    number = {
        "TeCP" : 0,
        "TeRP" : 0,
        "NoRE_TeP" : 0
    }

    for row in data:
        number[row["relation"]] += 1
    print(number)

    with open('train_bak.json', 'w') as outfile:
        json.dump(data, outfile)

    new_data = []

    for row in data:
        if(row["relation"] == "TeRP"):
            new_data.append(row)
        if(row["relation"] == "TeCP"):
            for _ in range(6):
                new_data.append(row)
        if(row["relation"] == "NoRE_TeP"):
            for _ in range(2):
                new_data.append(row)

    number = {
        "TeRP" : 0,
        "TeCP" : 0,
        "NoRE_TeP": 0
    }

    for row in new_data:
        number[row["relation"]] += 1

    random.shuffle(new_data)

    with open('train.json', 'w') as outfile:
        json.dump(new_data, outfile)

    print(number)
    print(sum(number.values()) / 3)

    os.chdir("../pp/data/")

    with open('train.json') as json_file:
        data = json.load(json_file)

    number = {
        "PIP" : 0,
        "NoRE_PIP" : 0
    }

    for row in data:
        number[row["relation"]] += 1
    print(number)

    with open('train_bak.json', 'w') as outfile:
        json.dump(data, outfile)

    new_data = []

    count = 0

    for row in data:
        if(row["relation"] == "PIP"):
            new_data.append(row)
        if(row["relation"] == "NoRE_PIP"):
            count += 1
            if(count % 6 == 0):
                new_data.append(row)

    number = {
        "PIP" : 0,
        "NoRE_PIP" : 0
    }

    for row in new_data:
        number[row["relation"]] += 1
    random.shuffle(new_data)

    with open('train.json', 'w') as outfile:
        json.dump(new_data, outfile)

    print(number)
    print(sum(number.values()) / 2)
    
if __name__ == '__main__':
    main()