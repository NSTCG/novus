#ai model building

import nltk
from nltk.stem.lancaster import LancasterStemmer
import nltk.stem.lancaster
import numpy as np

import tflearn
import pickle
import random
import json

stemmer=LancasterStemmer()
def load_pickle():
    import pickle
    with open("intents.pickle", "rb") as fi:
        data = pickle.load(fi)
    return data
data=load_pickle()
dataflag=0
if dataflag==0:
    try:

        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
        


    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        # print("\n \n\nwords (nltk.word_tokenize())(all words in each each hardcoded user input sentances)  :  " + str(words))
        # print(" \ndocs_x (array containing hard coded user input words ) :  " + str(docs_x))
        # print("\nlength of doc_x : " + str((len(docs_x))))
        # print("\ndocs_y  :  " + str(docs_y))
        # print("\nlabels (basically tag) :  " + str(labels))

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        # print("\n\nwords  lower stemmed :  " + str(words))

        words = sorted(list(set(words)))
        # print("words  lower stemmed and sorted :  " + str(words))
        # print(len(words))

        labels = sorted(labels)
        # print("labels sorted" + str(labels) + "\n \n \n")

        words = [stemmer.stem(tokenized_word.lower()) for tokenized_word in words]
        # print("new words"+ str(words))

        # one hot encoding
        # words= ['hello','a','buddy']
        # if the word is hello buddy
        # encoded = [1,0,1]           (whether the word is in the library /not .. if yes how many?

        training = []
        output = []

        # 1hotencoding for classes (greatings,goodbye ect) : [0 1] for goodbye
        out_empty = [0 for _ in range(len(labels))]
        # print("out empty (ie 0 for _ in words length) " + str(out_empty))
        # print(len(out_empty))

        for x, doc in enumerate(docs_x):
            # print("\n\n\nx,doc = " + str(x) + "," + str(doc))

            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]
            # rint("stemmed word no " + str(x) + " : " + str(wrds))

            for w in words:
                if w in wrds:

                    bag.append(1)
                else:
                    bag.append(0)
                # print("\nwrds " + str(w) + "   : " + str(wrds))
            # print("bag=" + str(bag))
            output_row = out_empty[:]

            # print("docs_y[x] : " + str(docs_y[x]))
            # print("labels.index(docs_y[x]) : " + str(labels.index(docs_y[x])))
            output_row[labels.index(docs_y[x])] = 1
            # print("output row" + str(output_row))

            training.append(bag)
            # print("training=" + str(training))
            output.append(output_row)
            # print("output" + str(output))

        training = np.array(training)
        output = np.array(output)
        # print("\ntrain " + str(x) + "   : " + str(training))
        # print("\noutput " + str(x) + "   : " + str(output))
        print(words,labels,training,output)
        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)
if dataflag==1:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #print("\n \n\nwords (nltk.word_tokenize())(all words in each each hardcoded user input sentances)  :  " + str(words))
    #print(" \ndocs_x (array containing hard coded user input words ) :  " + str(docs_x))
    #print("\nlength of doc_x : " + str((len(docs_x))))
    #print("\ndocs_y  :  " + str(docs_y))
    #print("\nlabels (basically tag) :  " + str(labels))

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    #print("\n\nwords  lower stemmed :  " + str(words))

    words = sorted(list(set(words)))
    #print("words  lower stemmed and sorted :  " + str(words))
    #print(len(words))

    labels = sorted(labels)
    #print("labels sorted" + str(labels) + "\n \n \n")

    words=[stemmer.stem(tokenized_word.lower()) for tokenized_word in words]
    #print("new words"+ str(words))

    # one hot encoding
    # words= ['hello','a','buddy']
    # if the word is hello buddy
    # encoded = [1,0,1]           (whether the word is in the library /not .. if yes how many?

    training = []
    output = []

    # 1hotencoding for classes (greatings,goodbye ect) : [0 1] for goodbye
    out_empty = [0 for _ in range(len(labels))]
    #print("out empty (ie 0 for _ in words length) " + str(out_empty))
    #print(len(out_empty))

    for x, doc in enumerate(docs_x):
        #print("\n\n\nx,doc = " + str(x) + "," + str(doc))

        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]
        #rint("stemmed word no " + str(x) + " : " + str(wrds))

        for w in words:
            if w in wrds:

                bag.append(1)
            else:
                bag.append(0)
            #print("\nwrds " + str(w) + "   : " + str(wrds))
        #print("bag=" + str(bag))
        output_row = out_empty[:]

        #print("docs_y[x] : " + str(docs_y[x]))
        #print("labels.index(docs_y[x]) : " + str(labels.index(docs_y[x])))
        output_row[labels.index(docs_y[x])] = 1
        #print("output row" + str(output_row))

        training.append(bag)
        #print("training=" + str(training))
        output.append(output_row)
        #print("output" + str(output))

    training = np.array(training)
    output = np.array(output)
    #print("\ntrain " + str(x) + "   : " + str(training))
    #print("\noutput " + str(x) + "   : " + str(output))

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)


net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net =tflearn.regression(net)
model=tflearn.DNN(net)


flag_train=0
if flag_train==0:
    model.load("model.tflearn")
    try:

        
        model2.load("model2.tflearn")

    except:
        #model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        #model.save("model.tflearn")
        #model2.fit(training2, output2, n_epoch=1000, batch_size=8, show_metric=True)
        #model2.save("model2.tflearn")
        print("exception error")
if flag_train==1:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    flag_train=0
def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=(1)
    return np.array(bag)



#funcions for flask

def predict(input):
    #print("chat")
    command="auto"
    inp = input
    #inp=tcglisten("en")
    # inp=input()

    #print("user : " + str(inp))
    if inp.lower == "quit":
        b=1
    # print([bag_of_words(inp,words)])
    results = model.predict([bag_of_words(inp, words)])[0]
    # print(results)
    results_index = np.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.1:

        for tg in data["intents"]:
            if tg['tag'] == tag:
                i = data["intents"].index(tg)
                responses = tg['responses']
                ran = random.choice(responses)
                j = responses.index(ran)
                name = "response_" + str(i) + "_" + str(j)
                if command == "auto":
                    file = 'audiobase\{}.mp3'.format(name)
                if command == "manual":
                    file = 'audiobase\{}.wav'.format(name)

                #from playsound import playsound
                #playsound(file)
                #print("nstcg : " + str(ran))
        return ran
    else:
        ran="i didnt get it"
        return ran

def predict2(input):
    #print("chat")
    command="auto"
    inp = input
    #inp=tcglisten("en")
    # inp=input()

    #print("user : " + str(inp))
    if inp.lower == "quit":
        b=1
    # print([bag_of_words(inp,words)])
    results = model2.predict([bag_of_words(inp, words)])[0]
    # print(results)
    results_index = np.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.1:

        for tg in data["intents"]:
            if tg['tag'] == tag:
                i = data["intents"].index(tg)
                responses = tg['responses']
                ran = random.choice(responses)
                j = responses.index(ran)
                name = "response_" + str(i) + "_" + str(j)
                if command == "auto":
                    file = 'audiobase\{}.mp3'.format(name)
                if command == "manual":
                    file = 'audiobase\{}.wav'.format(name)

                #from playsound import playsound
                #playsound(file)
                #print("nstcg : " + str(ran))
        return ran
    else:
        ran="i didnt get it"
        return ran
def energy_charge(units):
    if units<=500:
        units = units / 2
        a = []
        b = []
        num = int(units // 50)
        last = units % 50
        # print(num)
        # print(last)
        for i in range(num):
            a.append(50)
        a.append(last)
        # print(a)
        for j in range(1, len(a) + 1):
            if j == 1:  # <50
                bill1 = a[j - 1] * 3.15
                b.append(bill1)
            if j == 2:  # 50 to100
                bill2 = a[j - 1] * 3.70
                b.append(bill2)
            if j == 3:  # 100 to 150
                bill3 = a[j - 1] * 4.80
                b.append(bill3)
            if j == 4:  # 150 to 200
                bill4 = a[j - 1] * 6.40
                b.append(bill4)
            if j == 5:  # 200 to 250
                bill5 = a[j - 1] * 7.60
                b.append(bill5)
        # print(b)
        return 2 * np.sum(b)
    if units>500 and units<=600:
        return (units*5.8)
    elif units > 600 and units<=700:
        return (units* 6.6)
    elif units > 700 and units<=800:
        return (units* 6.9)
    elif units > 800 and units<=1000:
        return (units * 7.1)
    elif units > 1000:
        return (units * 7.9)
def fc(units):
    if units >0 and units <= 100:
        return 70,40
    elif units>100 and units<=200:
        return 90,40
    elif units >200 and units<=240:
        return 110,40
    elif units >240 and units<=300:
        return 110,0
    elif units >300 and units<=400:
        return 140,0
    elif units >400 and units<=500:
        return 160,0
    elif units >500 and units<=600:
        return 200,0
    elif units >600 and units<=700:
        return 220,0
    elif units >700 and units<=800:
        return 240,0
    elif units >800 and units<=1000:
        return 260,0
    elif units >1000 :
        return 300,0
def translator(text,src_in,dest_in):
    from googletrans import Translator
    trans = Translator()
    t = trans.translate(
        text,src=src_in,dest=dest_in
    )
    #print(f'Source:{t.src}')
    #print(f'Destination:{t.dest}')
    return (f'{t.text}')
response="blah"
#the flask

from flask import Flask,url_for,render_template,request,redirect

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
    #return f"<h1>Welcome iam you personalised ai chatbot: NSTCG mark 1, you can ask me anything by putting a slash (/) and the question in the browser</h1>"
@app.route("/<usr>")
def user(usr):
    #res=predict(usr)
    return f"<h1>{usr}</h1>"
@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        response=predict(user)
        response1="You : " + str(user)
        response2="NSTCCG : " + str(response)
        render_template("login.html", res1=response1,res2=response2,respure=response)
        #print(user)
        return render_template("login.html", res1=response1,res2=response2,tospeak=response)
    else:
        return render_template("login.html")
@app.route("/babu", methods=["POST", "GET"])
def babu():
    if request.method == "POST":
        try:
            user = request.form["nm"]
        except:
            user = request.form["myimage"]


        response=predict(user)
        response1="You : " + str(user)
        response2="NSTCCG : " + str(response)
        render_template("babu.html", res1=response1,res2=response2,respure=response)
        #print(user)
        return render_template("babu.html", res1=response1,res2=response2)

    else:
        return render_template("babu.html")
@app.route("/update", methods=["POST", "GET"])
def update():
    if request.method == "POST":
        user = request.form["nm"]
        user2 = request.form["nm2"]
        #response=predict(user)
        response1=user
        response2=user2
        render_template("update.html", res1=response1,res2=response2)
        #print(user)
        return render_template("update.html", res1=response1,res2=response2)
    else:
        return render_template("update.html")
@app.route("/energycharge", methods=["POST", "GET"])
def energycharge():
    if request.method == "POST":
        units = request.form["nm"]
        bill=energy_charge(float(units))
        if int(units)>500:
            response2="The bill is non-telescopic"
        else:
            response2="The bill is telescopic"
        response1="Your estimated  bill : " + str(bill) + " rs"
        #print(user)
        fcc,fcsub=fc(float(units))
        return render_template("energycharge.html",units="Units consumed : " + units, res1=response1,res2=response2,fcc="fixed charge : " + str(fcc) ,fcsub= "fixed charge subsidy" + str(fcsub))
    else:
        return render_template("energycharge.html")

@app.route("/recognise",methods=["POST","GET"])
def recognise():
    if request.method == "POST":
        user = request.form["nm2"]
        response=predict(user)
        response1="You : " + str(user)
        response2="NSTCCG : " + str(response)
        #print(user)
        return render_template("ui.html", res1=response1,res2=response2,tospeak=response)
    else:
        word="hello holy world"
        return render_template("ui.html",tospeak=word)

    word="hello holy world"
    return render_template("ui.html",z=word)
    #return f"<h1>Welcome iam you personalised ai chatbot: NSTCG mark 1, you can ask me anything by putting a slash (/) and the question in the browser</h1>"



@app.route("/pm",methods=['POST'])                      #predict model by ajax
def pm():
    data=request.form
    questiontemp=data["text"]
    language=data["lang"]
    if language=="en":
        question=questiontemp
    else:
        question=translator(questiontemp,language,"en")
    

    answer=predict(question)
    a={}
    a["answer"]=answer
    a["question"]=question
    print(a)
    return a

if __name__ == "__main__":
    app.run(use_reloader=False,debug=True, host='0.0.0.0', port=80)



#for loading css : <link rel=""stylesheet" href="{{url_for('static', filename='footer.css')}}>
#for importing image :  <p><img src="https://drive.google.com/uc?export=view&id=1Q1dJLHK5L_zpSgf0yWpXUmp4Mcyrlfho"></p>

#<div class="w3-container">
#     <h1 class="w3-animate-fading">Animation is Fun!</h1>
#    </div>