from fun import *

def main():

    tic = time.perf_counter()

    import warnings
    warnings.filterwarnings('ignore')

    # PARAMETERS
    # in a web application these could be entered or chosen from menu
    # file ids here as a list of integers. If only one, enter as [nn].

    # parameters:
    # [1] output target: f=file, s=screen (default)
    # [2] find top level term: 1=yes, 0=no (default)
    # [3] model:
    #   0='biobert'
    #   1='en_core_sci_md'
    #   2='en_core_sci_lg'
    #   3='en_core_sci_scibert'
    #   4='en_ner_craft_md'
    #   5='en_ner_jnlpba_md'
    #   6='en_ner_bc5cdr_md' (default)
    #   7=en_ner_bionlp13cg_md'
    # [4] process title only: 1=yes, 0=no (default)
    # [5] number of columns: 2 | 1 (default)
    # [6] multiple articles: 1=yes, 0=no (default)

    # use screen for output? True
    # sys.argv[1] 'S'=True; else: False
    try:
        if sys.argv[1].upper()=='S':
            screen=True
        elif sys.argv[1].upper()=='F':
            screen=False
        else:
            screen=True
    except:
        screen=True
    if not screen:
    # initialize output file
        with open('terms.txt', 'w') as f:
            f.write('\n')

    # top level terms? True|False
    # sys.argv[2] 1=True; 2=False
    try:
        if sys.argv[2]=='1':
            TLT = True
        elif sys.argv[2]=='0':
            TLT = False
        else:
            TLT = False
    except:
        TLT = False

    # model? 
    # sys.argv[3]
    try:
        if sys.argv[3] == '0':
            model = 'biobert'
        elif sys.argv[3] == '1':
            model = 'en_core_sci_md'
        elif sys.argv[3] == '2':
            model = 'en_core_sci_lg'
        elif sys.argv[3] == '3':
            model = 'en_core_sci_scibert'
        elif sys.argv[3] == '4':
            model = 'en_ner_craft_md'
        elif sys.argv[3] == '5':
            model = 'en_ner_jnlpba_md'
        elif sys.argv[3] == '6':
            model = 'en_ner_bc5cdr_md'
        elif sys.argv[3] == '7':
            model = 'en_ner_bionlp13cg_md'
        else:
            model = 'en_ner_bc5cdr_md'
    except:
        model = 'en_ner_bc5cdr_md'

    # process the title only? True|False
    # sys.argv[4] 1=True; 0=False (default)
    try:
        if sys.argv[4] == '1':
            title_only = True
        elif sys.argv[4] == '0':
            title_only = False
        else:
            title_only = False
    except:
        title_only = False

    # number of columns?
    # sys.argv[5] 2 | 1 (default)
    try:
        if sys.argv[5]=='1':
            columns=False
        elif sys.argv[5]=='2':
            columns=True
        else:
            columns = False
    except:
        columns = False

    # multiple articles? True|False
    # sys.argv[6] 1=True; 0=False (default)
    try:
        if sys.argv[6]=='1':
            multiple_articles=True
        elif sys.argv[6]=='0':
            multiple_articles=False
        else:
            multiple_articles=False
    except:
        multiple_articles = False

    # select vocabulary
    # sys.argv[7] type=string
    try:
        vocab=sys.argv[7]
    except:
        vocab = 'MSH'
    # find terms?
    AT = True
    BT = False # iff AT = True
    Roots = False # iff both AT and BT are True
    CUI = False # iff AT = True

    # create empty dataframe 
    columns = ['id', 'filename', 'text', 'text_expanded', 'entities', 'entities_text', 'mesh_terms']
    df = pd.DataFrame(columns = columns)

    # read list of pdfs
    pdfs = [file for file in os.listdir('pdf') if file[-4:]=='.pdf']

    # load existing tags dataframe
    # read CSV files
    tags1_df = pd.read_csv('tags/cmes_tags.csv')
    tags2_df = pd.read_csv('tags/additional_tags.csv')
    tags_df = pd.concat([tags1_df[['Topic_ID', 'Tag_Name']], tags2_df[['Topic_ID', 'Tag_Name']]], axis=0)
    tags_df = tags_df.drop_duplicates()
    tags_df = tags_df.groupby(['Topic_ID'])['Tag_Name'].apply(list).reset_index(name='Tag_Names')

    terms_out = []
    output = []

    # loop over files to be processed
    loop_no = 0
    for f in pdfs:

        # extract file id from the filename
        id = f[:4]

        # convert pdf to list of lists (pages of lines)
        text = pdf_to_text('pdf/'+f)

        # remove extraneous lines
        # returns list (pages) of lists (lines)
        text = remove_lines(text)
        if columns:
            # merge columns
            # returns list (pages) of lists (lines)
            text = two_cols_to_one(text)

        # split document into articles
        if multiple_articles:
            # returns list of articles
            titles, authors, texts = split_articles(text)
        else:
            # returns simple list (one article)
            text = [line.strip() for page in text for line in page]
            titles = []
            authors = []
            for i, line in enumerate(text):
                if re.search(r'(MD|DO)', line):
                    titles.append(' '.join(text[:i]))
                    authors.append(text[i])
                    break
            texts = ['\n'.join(text)]

        taglist = tags_df['Tag_Names'].drop_duplicates().tolist()
        taglist = list(set([tag for tags in taglist for tag in tags]))
        taglist = sorted([tag for tag in taglist if tag==tag]) # remove nan
        
        try:
            tags = tags_df.loc[tags_df['Topic_ID'] == int(id)]['Tag_Names'].values[0]
        except:
            tags = []

        if tags == ['N/A']:
            tags = []

        if screen:
            print("Number of articles:", len(texts))
        for i in range(len(texts)):
      
            loop_no += 1
            print('loop number', loop_no)

            output_dict = {}

            if title_only:
                try:
                    text = titles[i]
                except:
                    text = ''
            else:
                text = texts[i]

            if screen:
                print('==========', i+1, '==========')
                print('File ID:', id)
                try:
                    print('Title:', colored(titles[i], 'cyan'))
                except:
                    print('Title:', 'No title found.')
                try:
                    print('Authors:', colored(authors[i], 'cyan'))
                except:
                    print('Authors:', 'No authors found.')
            else:
                with open('terms.txt', 'a') as f:
                    f.write('\n')
                    f.write('====================================================='+'\n')
                    f.write('File ID: '+str(id)+'\n')
                    f.write('====================================================='+'\n')
                    try:
                        f.write('Title: '+titles[i]+'\n')
                        output_dict['title'] = titles[i]
                    except:
                        f.write('Title: No title found.'+'\n')
                        output_dict['title'] = ''
                    try:
                        f.write('Authors: '+authors[i]+'\n')
                    except:
                        f.write('Authors: No authors found.'+'\n')

                fn, url = '', ''
                output_dict['id'] = id
#               for filename in filenames:
#                    if id == filename[0]:
#                   fn = filename[1]
#                   url = filename[2]
#               output_dict['filename'] = fn
#               output_dict['url'] = url
            
            if model == 'biobert':
                textlines = text.replace('"', '\"').split('\n')
                input_text = '"' + '", "'.join(textlines) + '"' 
                output_dict = Multi_Label_Classification_of_Pubmed_Articles(input_text)
                sorted_dict = dict(sorted(output_dict.items(), key=lambda x:x[1], reverse=True))
                my_table = PrettyTable()
                for item in sorted_dict.items():
                    my_table.add_row([item[0], '{:.1%}'.format(item[1])])
                print(my_table)
                
            elif AT:
                print('tags:', tags)
                x, nlp, doc, linker, entity_dict, data_df, output_dict = get_umls_terms(text, tags, screen, vocab, BT, Roots, CUI, TLT, output_dict, model)

            #print(entity_dict)

            output.append(output_dict)

        #x.insert(0, f)
        #x.insert(0, id)
        #terms_out.append(x)

    toc = time.perf_counter()
    print(f"Elapsed time: {toc - tic:0.1f} seconds.")


if __name__ == '__main__':
    main()
