import re
import sys
import os
import time
from termcolor import colored
import json
from lxml.html import fromstring
import lxml.html as lh
import requests
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from spacy import displacy
import en_ner_bionlp13cg_md
import en_ner_jnlpba_md
import en_ner_bc5cdr_md
import en_ner_craft_md
import en_core_sci_lg
import en_core_sci_md
import en_core_sci_sm
import spacy
import scispacy
import contractions
import pandas as pd
import warnings
import transformers
import torch
import numpy as np
from prettytable import PrettyTable
from typing import Dict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
warnings.filterwarnings('ignore')

num_labels = 14
# Loading Model and Tokenizer from Hugging Face Spaces
model = BertForSequenceClassification.from_pretrained(
    "owaiskha9654/Multi-Label-Classification-of-PubMed-Articles", num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(
    'owaiskha9654/Multi-Label-Classification-of-PubMed-Articles', do_lower_case=True)



# Core models

# NER specific models

# Tools for extracting & displaying data


# clear screen for Windows, Mac, Linux OS

def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

# convert PDF; split into pages and lines; returns list of lists (lines in pages)
# def pdf_to_text(filename):
#    with open(filename, 'rb') as f:
#        doc = [page.split('\n') for page in pdftotext.PDF(f)]
#    return doc


def pdf_to_text(filename):
    command = "pdftotext -layout -eol 'unix' '"+filename+"' 'out.txt'"
    os.system(command)
    pages = open("out.txt", "r").read().split('\f')
    doc = []
    for page in pages:
        lines = page.split('\n')
        doc.append(lines)
    return doc

# remove unnecessary lines


def remove_lines(doc):
  newdoc = []
  for page in doc:
    newpage = []
    for line in page:
      if line.lstrip()[:15].lower() == 'editor-in-chief':
        newpage.append('')
      elif line.lstrip()[:15].lower() == 'editor in chief':
        newpage.append('')
      elif line.lstrip()[:16].lower() == 'executive editor':
        newpage.append('')
      elif line.lstrip()[:15].lower() == 'managing editor':
        newpage.append('')
      elif line.lstrip()[:25].lower() == 'associate managing editor':
        newpage.append('')
      elif line.lstrip()[:12].lower() == 'pcma editors':
        newpage.append('')
      elif line.lstrip()[:16].lower() == 'associate editor':
        newpage.append('')
      elif line.lstrip()[:12].lower() == 'print editor':
        newpage.append('')
      elif line.lstrip()[:15].lower() == 'abstract editor':
        newpage.append('')
      # elif line.strip() == '':
      #  pass
      elif re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s20\d+.*?Volume\s+\d+.*?Issue\s+\d+', line.strip()):
        newpage.append('')
      elif re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s\d\d\d\d\s|\sAbstract', line.strip()):
        newpage.append('')
      elif re.search(r'^[0-9]+$', line.strip()):
        newpage.append('')
      elif re.search(r'^\d+\s+em:rap written summary.*?www.emrap.org\s*?$', line.strip().lower()):
        newpage.append('')
      elif re.search(r'emrap.org', line.strip().lower()):
        newpage.append('')
      elif re.search(r'primary care medical abstracts', line.strip().lower()):
        newpage.append('')
      elif re.search(r'right on prime', line.strip().lower()):
        newpage.append('')
      elif line.strip().lower() == 'notes':
        newpage.append('')
      else:
        newpage.append(line)
    if newpage == []:
      pass
    else:
      newdoc.append(newpage)
  return newdoc


def display(message, text):
    scroll = False
    for page in text:
        for line in page:
            print(line)
        if not scroll:
            response = input(
                '\n'+message+'\n'+'Press the X key to stop printing; S key for continuous scrolling; any other key to continue... '+'\n')
            if response.lower() == 'x':
                break
            elif response.lower() == 's':
                scroll = True

# find max midpoint of lines


def line_midpoint(page):
    length = 0
    for line in page:
        if len(line) > length:
            length = len(line)
    return length // 2

# split lines by finding spaces near midpoint


def split_line(line, mid):
    try:
        match = re.search('\s{2}(?=\S)', line[mid-20:mid+20])
        if match:
            # return tuple with line split
            return (line[:mid-25+match.span()[0]].strip(), line[mid-25+match.span()[1]:].strip())
        else:
            # nothing in right column
            return (line.strip(), '')
    except:
        # nothing in right column
        return (line.strip(), '')

# rebuild document in continuous column


def two_cols_to_one(doc):
    newdoc = []
    for page in doc:
        leftcol = []
        rightcol = []
        # locate middle of line
        mid = line_midpoint(page)
        for line in page:
            # split line at space near midpoint
            split = split_line(line, mid)
            if split[0] != '':
                # add to left column
                leftcol.append(split[0])
            if split[1] != '':
                # add to right column
                rightcol.append(split[1])
        # concatenate columns
        newpage = leftcol + rightcol
        newdoc.append(newpage)
    return newdoc

# split document into articles


def split_articles(doc):
    split_text = []
    title_pos = []
    titles = []
    authors = []
    p1 = -1
#    for page in doc:
#        for line in page:
#            print(colored(line, 'cyan'))
    text = [line for page in doc for line in page]
    for i, line in enumerate(text):
        if re.search(r'(MD|DO)', line):
            p2 = i-1
            titles.append(text[i-1])
            authors.append(text[i])
            if p1 != -1:
                title_pos.append((p1, p2))
            p1 = i+1
    title_pos.append((p1, i+1))
    for pos in title_pos:
        article_text = []
        for line in text[pos[0]: pos[1]]:
            article_text.append(line)
        split_text.append(article_text)
    return titles, authors, ['\n'.join(article) for article in split_text]


def contraction_expansion(text):
    expanded_word = []
    for word in text.split():
        # using contractions.fix to expand
        expanded_word.append(contractions.fix(word))
    return ' '.join(expanded_word)


def preprocessing(filename, text):
    id = filename[:re.search('_', filename).span()[0]]
    lst = [[id, filename, article] for article in text]
    df = pd.DataFrame(lst, columns=['id', 'filename', 'text'])
    df['text_expanded'] = df['text'].apply(contraction_expansion)
    return df


def entity_extraction(text):
    doc = nlp(text)
    return list(doc.ents)


def list_to_string(lst):
    lst = [str(term) for term in lst]
    return ' '.join(lst)


def get_umls_terms(text, tags, screen, vocab, BT, Roots, CUI, TLT, output_dict, model):
    #    models = ['en_core_sci_md','en_core_sci_lg','en_core_sci_scibert','en_ner_craft_md','en_ner_jnlpba_md','en_ner_bc5cdr_md','en_ner_bionlp13cg_md']
    #    model = 'en_ner_bc5cdr_md'
    first = True
    entities = []
    nlp = spacy.load(model)
    nlp.add_pipe("abbreviation_detector")
    # linker_name values: 'umls'|'mesh'
    nlp.add_pipe("scispacy_linker", config={
                 "resolve_abbreviations": True, "linker_name": "mesh"})
    doc = nlp(text)
    if screen:
        print("Abbreviations:")
    else:
        with open('terms.txt', 'a') as f:
            f.write("Abbreviations:"+'\n')
    abbreviations = []
    for abrv in doc._.abbreviations:
        abbreviations.append((abrv.text, abrv._.long_form))
    abbreviations = list(set(abbreviations))
    if len(abbreviations) == 0:
        if screen:
            print(colored('NONE FOUND', 'cyan'))
        else:
            with open('terms.txt', 'a') as f:
                f.write('NONE FOUND'+'\n')
            output_dict['abbreviations'] = [()]
    else:
        abv = []
        for abbreviation in abbreviations:
            if screen:
                print(
                    colored(f"{abbreviation[0]} \t {abbreviation[1]}", 'cyan'))
            else:
                with open('terms.txt', 'a') as f:
                    f.write(f"{abbreviation[0]} \t {abbreviation[1]}"+'\n')
                abv.append((f"{abbreviation[0]}", f"{abbreviation[1]}"))
        output_dict['abbreviations'] = abv
    umls_list, tag_list = [], []
    if screen:
        print('Existing tags:')
    else:
        with open('terms.txt', 'a') as f:
            f.write('Existing tags:'+'\n')
    output_dict['tags'] = []
#    if screen:
#        print('tags:', tags)
    if tags == []:
        if screen:
            print(colored('NONE', 'cyan'))
        else:
            with open('terms.txt', 'a') as f:
                f.write('NONE'+'\n')
    else:
        tag_string = ', '.join([tag for tag in tags if tag == tag])
#        if screen:
#            print('tag_string:', tag_string)
#            print('tags:', tags)
        tag_list = tag_string.split(', ')
        for tag in tag_list:
            if screen:
                print(colored(tag, 'cyan'))
            else:
                with open('terms.txt', 'a') as f:
                    f.write(tag+'\n')
                output_dict['tags'].append(tag)
    umls_list.append(tag_list)
    if screen:
        print('Suggested terms:')
    else:
        with open('terms.txt', 'a') as f:
            f.write('Suggested terms:'+'\n')
    linker = nlp.get_pipe("scispacy_linker")
    terms = []
    data = []
    for ent in linker(doc).ents:
        entity = ent.text
        label = ent.label_
        for umls_ent in ent._.kb_ents:
            if float(umls_ent[1]) == 1.0:
                cui = umls_ent[0]
                umls_term = linker.kb.cui_to_entity[umls_ent[0]].canonical_name
                terms.append((cui, umls_term))
        entities.append((entity, label))

        # NER data
        show_only_top = True
        for ent_id, score in ent._.umls_ents:
            kb_entity = linker.umls.cui_to_entity[ent_id]
            tuis = ",".join(kb_entity.types)
            data.append([
                ent.text.lower().replace('\n', ' ').replace('  ', ' '),
                kb_entity.canonical_name,
                ent_id,
                tuis,
                score,
                ent.start,
                ent.end,
            ])
            if show_only_top:
                break

#    if screen:
#        print('entity\tMeSH term\tMesh ID\tscore\tbegin\tend')
#        for line in data:
#            print(line[0]+'\t'+line[1]+'\t'+line[2]+'\t'+str(line[4])+'\t'+str(line[5])+'\t'+str(line[6]))

    attrs = ["text", "Canonical Name", "Concept ID",
             "TUI(s)", "Score", "start", "end"]
    df = pd.DataFrame(data, columns=attrs)
    output_dict['terms'] = []
    if df.empty:
        if screen:
            print(colored('NONE', 'cyan'))
        else:
            with open('terms.txt', 'a') as f:
                f.write('NONE'+'\n')
    else:
        df = df.groupby(['Canonical Name', 'Concept ID']).agg(
            {'Score': ['mean', 'count']})
        # df.columns = df.columns.get_level_values(1)
        df = df.sort_values(
            by=[('Score', 'count'), ('Score', 'mean')], ascending=False)
        df2 = df[(df[('Score', 'mean')] >= 0.8) & (df[('Score', 'count')] > 1)]
        if len(df2) < 3:
            df = df.head(2)
        else:
            df = df2

        if screen:
            print(colored(df, 'cyan'))
        else:
            with open('terms.txt', 'a') as f:
                f.write('Canonical Name\tConcept ID\tmean score\tterm count'+'\n')
                for index, row in df.iterrows():
                    f.write(index[0]+'\t'+index[1]+'\t'+str(
                        round(row[(('Score', 'mean'))], 2))+'\t'+str(int(row[('Score', 'count')]))+'\n')
                    output_dict['terms'].append(index[0])

    entity_dict = {entity: entities.count(entity) for entity in entities}
    sorted_entity_dict = {key: value for key, value in sorted(
        entity_dict.items(), key=lambda item: (item[1], item[0]), reverse=True)}
    terms = list(set(terms))
    if terms == []:
        if screen:
            print(colored('NONE', 'cyan'))
        else:
            with open('terms.txt', 'a') as f:
                f.write('NONE'+'\n')
    else:
        for term in terms:
            if screen:
                # print(colored(term[0]+' '+term[1], 'cyan'))
                if BT:
                    parent_list, id_list = [], []
                    for term_tuple in walk_hierarchy(term[0], 'parents'):
                        parent_list.append(term_tuple[0])
                        id_list.append(term_tuple[1])
                    # parents = '; '.join(parent_list)
                    for parent in parent_list:
                        print(' '*9+'BT  '+colored(parent, 'cyan'))
                    if Roots:
                        # root_terms = '; '.join(get_roots(id_list))
                        for root in get_roots(id_list):
                            print(' '*9+'Root:  '+colored(root, 'cyan'))
    umls_list.append([term[1] for term in terms])

    output_dict['top'] = ''
    if TLT:
        if screen:
            print('Top level term:')
        else:
            with open('terms.txt', 'a') as f:
                f.write('Top level term:'+'\n')
        if screen:
            try:
                print(
                    colored(get_roots([df.index.values.tolist()[0][1]])[0], 'cyan'))
            except:
                print(colored('NONE', 'cyan'))
        else:
            with open('terms.txt', 'a') as f:
                try:
                    f.write(
                        get_roots([df.index.values.tolist()[0][1]])[0]+'\n')
                    output_dict['top'] = get_roots(
                        [df.index.values.tolist()[0][1]])[0]
                except:
                    f.write('NONE'+'\n')
        # for tlt in (get_roots([term[1] for term in df[df[('Score', 'count')] > 4].index.values.tolist()])):
        #    print(tlt)

    return ['\n'.join(terms) for terms in umls_list], nlp, doc, linker, sorted_entity_dict, df, output_dict

# get ticket


def gettgt():
    with open('apikey.txt') as f:
        apikey = f.readline().strip()
    uri = 'https://utslogin.nlm.nih.gov'
    auth_endpoint = "/cas/v1/api-key"
    params = {'apikey': apikey}
    h = {"Content-type": "application/x-www-form-urlencoded",
         "Accept": "text/plain", "User-Agent": "python"}
    r = requests.post(uri+auth_endpoint, data=params, headers=h)
    response = fromstring(r.text)
    tgt = response.xpath('//form/@action')[0]
    return tgt


def getst(tgt):
    params = {'service': 'http://umlsks.nlm.nih.gov'}
    h = {'Content-type': 'application/x-www-form-urlencoded',
         'Accept': 'text/plain', 'User-Agent': 'python'}
    r = requests.post(tgt, data=params, headers=h)
    st = r.text
    return st


def umls_search(string):
    # UMLS/MeSH search function
    # string = search words
    # searchType = words | exact | etc.
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/current"
    tgt = gettgt()
    vocab = 'MSH'
    returnIdType = 'sourceUi'
    pageNumber = 0
    terms = []
    for i, word in enumerate(string.split()):
        # search first 3 terms only
        if i > 0:
            break
        while True:
            pageNumber += 1
            ticket = getst(tgt)
            query = {'string': word, 'ticket': ticket, 'searchType': 'words',
                     'returnIdType': returnIdType, 'pageNumber': pageNumber, 'sabs': vocab}
            r = requests.get(uri+content_endpoint, params=query)
            r.encoding = 'utf-8'
            items = json.loads(r.text)
            jsonData = items["result"]
            for result in jsonData["results"]:
                if jsonData["results"][0]["ui"] != "NONE":
                    terms.append((result["name"], result["ui"]))
                    print(result['name'], result['ui'])
                    break
            if jsonData["results"][0]["ui"] == "NONE":
                break
    if terms == []:
        return ''
    else:
        # only returning first term
        return terms[0]


def walk_hierarchy(identifier, operation):
    # identifier = MeSH id
    # operation = 'atoms' | 'parents' | 'children' | 'ancestors' | 'descendants' | 'relations'
    source = 'MSH'
    uri = 'https://uts-ws.nlm.nih.gov'
    content_endpoint = '/rest/content/current/source/' + \
        source+'/'+identifier+'/'+operation
    # get ticket for session
    tgt = gettgt()
    pageNumber = 1
    terms = []
    while True:
        query = {'ticket': getst(tgt), 'pageNumber': pageNumber}
        r = requests.get(uri+content_endpoint, params=query)
        r.encoding = 'utf-8'
        items = json.loads(r.text)
        pageCount = items['pageCount']
        for result in items['result']:
            terms.append((result['name'], result['ui']))
        pageNumber += 1
        if pageNumber > pageCount:
            break
    return terms

# get roots

def get_roots(identifiers):
    mesh_categories = []
    for identifier in identifiers:
        parents = walk_hierarchy(identifier, 'parents')
        for parent in parents:
            test = parent
            while True:
                try:
                    t = walk_hierarchy(test[1], 'parents')[0][1][0]
                    if t != 'D':  # i.e. not a MeSH term
                        mesh_categories.append(test[0])
                        # mesh_categories.append(walk_hierarchy(test[1], 'parents')[0][0])
                        break
                    else:
                        test = walk_hierarchy(test[1], 'parents')[0]
                except:
                    break
    return list(set(mesh_categories))

# Source: https://huggingface.co/spaces/owaiskha9654/Multi-Label-Classification-of-Pubmed-Articles/blob/main/app.py
# This wrapper function will pass the article into the model
def Multi_Label_Classification_of_Pubmed_Articles(model_input: str) -> Dict[str, float]:
    dict_custom = {}
    # splitting inputext into 2 parts
    Preprocess_part1 = model_input[:len(model_input)]
    Preprocess_part2 = model_input[len(model_input):]
    dict1 = tokenizer.encode_plus(
        Preprocess_part1, max_length=1024, padding=True, truncation=True)
    dict2 = tokenizer.encode_plus(
        Preprocess_part2, max_length=1024, padding=True, truncation=True)

    dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
    dict_custom['token_type_ids'] = [
        dict1['token_type_ids'], dict1['token_type_ids']]
    dict_custom['attention_mask'] = [
        dict1['attention_mask'], dict1['attention_mask']]

    outs = model(torch.tensor(dict_custom['input_ids']), token_type_ids=None,
                 attention_mask=torch.tensor(dict_custom['attention_mask']))
    b_logit_pred = outs[0]
    pred_label = torch.sigmoid(b_logit_pred)

    ret = {
        "Anatomy [A]": float(pred_label[0][0]),
        "Organisms [B]": float(pred_label[0][1]),
        "Diseases [C]": float(pred_label[0][2]),
        "Chemicals and Drugs [D]": float(pred_label[0][3]),
        "Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]": float(pred_label[0][4]),
        "Psychiatry and Psychology [F]": float(pred_label[0][5]),
        "Phenomena and Processes [G]": float(pred_label[0][6]),
        "Disciplines and Occupations [H]": float(pred_label[0][7]),
        "Anthropology, Education, Sociology, and Social Phenomena [I]": float(pred_label[0][8]),
        "Technology, Industry, and Agriculture [J]": float(pred_label[0][9]),
        "Information Science [L]": float(pred_label[0][10]),
        "Named Groups [M]": float(pred_label[0][11]),
        "Health Care [N]": float(pred_label[0][12]),
        "Geographicals [Z]": float(pred_label[0][13])}
    return ret

# Multi_Label_Classification_of_Pubmed_Articles modified
def classifier(model_input: str) -> Dict[str, float]:
    dict_custom = {}
    # splitting inputext into 2 parts
    Preprocess_part1 = model_input[:divmod(len(model_input), 2)[0]]
    Preprocess_part2 = model_input[divmod(len(model_input), 2)[0]:]
    dict1 = tokenizer.encode_plus(
        Preprocess_part1, max_length=1024, padding=True, truncation=True)
    dict2 = tokenizer.encode_plus(
        Preprocess_part2, max_length=1024, padding=True, truncation=True)

    dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
    dict_custom['token_type_ids'] = [
        dict1['token_type_ids'], dict1['token_type_ids']]
    dict_custom['attention_mask'] = [
        dict1['attention_mask'], dict1['attention_mask']]

    outs = model(torch.tensor(dict_custom['input_ids']), token_type_ids=None,
                 attention_mask=torch.tensor(dict_custom['attention_mask']))
    b_logit_pred = outs[0]
    pred_label = torch.sigmoid(b_logit_pred)

    ret = {
        "Anatomy [A]": float(pred_label[0][0]),
        "Organisms [B]": float(pred_label[0][1]),
        "Diseases [C]": float(pred_label[0][2]),
        "Chemicals and Drugs [D]": float(pred_label[0][3]),
        "Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]": float(pred_label[0][4]),
        "Psychiatry and Psychology [F]": float(pred_label[0][5]),
        "Phenomena and Processes [G]": float(pred_label[0][6]),
        "Disciplines and Occupations [H]": float(pred_label[0][7]),
        "Anthropology, Education, Sociology, and Social Phenomena [I]": float(pred_label[0][8]),
        "Technology, Industry, and Agriculture [J]": float(pred_label[0][9]),
        "Information Science [L]": float(pred_label[0][10]),
        "Named Groups [M]": float(pred_label[0][11]),
        "Health Care [N]": float(pred_label[0][12]),
        "Geographicals [Z]": float(pred_label[0][13])}
    return ret
