from django.shortcuts import render
import pandas as pd
import pickle

category_data = pd.read_csv("idx2category.csv")
idx2category = {row.k: row.v for idx, row in category_data.iterrows()}

with open('nb.pickle', mode="rb") as ff:
    model = pickle.load(ff)

def index(request):
  if request.method == "GET":
    return render(request, "nlp/home.html")
  else:
    title = [request.POST["title"]]
    result = model.predict(title)[0]
    pred = idx2category[result]
    return render(request, "nlp/home.html", {'title':pred})