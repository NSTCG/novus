import os,json,pickle

#load data from json

file=open('training.json',encoding='UTF-8')
data=json.load(file)
print(data)
file.close

#save a data to .txt
file=open('training.txt',"w",encoding='UTF-8')
file.write(str(data))
file.close

with open("intents.pickle","wb") as f:
    pickle.dump(data,f)