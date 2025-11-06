import csv
import json, re

def title_abstract_split(paper):

    titile_first_sentence = paper.split('. ')[0]
    words = re.findall(r'\b[A-Z][a-z]*\b', titile_first_sentence)
    if len(words)==2 and "BACKGROUND:" not in titile_first_sentence and "SUMMARY:" not in paper:
        title = titile_first_sentence.split(words[-1])[0]
    elif len(words) == 2 and "BACKGROUND:" in titile_first_sentence and "SUMMARY:" not in paper:
        title = titile_first_sentence.split("BACKGROUND:")[0]

    elif len(words)>2 and words.count(words[-1]) == 1 and "SUMMARY:" not in paper:
        title = titile_first_sentence.split(words[-1]+" ")[0]

    elif len(words)>2 and words.count(words[-1]) > 1 and "SUMMARY:" not in paper:
            abstract1 = words[-1]+titile_first_sentence.split(words[-1]+' ')[-1]
            title=titile_first_sentence.replace(abstract1,'')
    elif len(words)<=1 and "SUMMARY:" not in paper:
        title = titile_first_sentence.split('.')[0]
    elif "SUMMARY:" in paper:
        title = paper.split("SUMMARY:")[0]
    abstract = paper.replace(title,'')

    if " " not in abstract.split('.')[0]:
        first_word=abstract.split('.')[0]
        abstract=abstract.replace(first_word+".",'')
        title=title+first_word
        print(title)
        print(abstract)
    return title, abstract
articles_reader=csv.reader(open("articles.tsv",encoding="ISO-8859-1"))
articles = {}
for row in articles_reader:
    sentence = ''
    for sent in row:
        sentence = sentence+" "+sent
    id=row[0].split('\t')[0]
    articles["N"+str(id)]=sentence.replace(id+'\t','')

csv_reader=csv.reader(open("test.tsv"))
dump_data={}
for row in csv_reader:
    user_id, click, imp, label = row[0].split("\t")  #train
    id, user_id, click, imp = row[0].split("\t")  #test
    click_papers = click.strip().split()
    papers='COLLECTED PAPERS: \n[\n'
    for click_paper in click_papers:
        print(click_paper)
        content = articles[click_paper]
        print("content::::::::::::::", content)
        if len(content.split())>20:
            ti, ab = title_abstract_split(content)
        else:
            ti = content
            ab = "None"
        print(ti)
        print(ab)
        papers = papers + "{\"title\": " +"\""+ti+"\", "+"\"abstract\": \""+ab+"\"}\n"


    dump_data["prompt_"+str(user_id)]=papers+"\n]"
with open("./test_user_prompts.json", 'w', encoding="utf8") as f:

        json.dump(dump_data,f)
        f.close()



