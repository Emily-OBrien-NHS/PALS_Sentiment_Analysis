import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string
import calendar
from transformers import pipeline
import spacy
import time
import ast
import re
import os
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          pipeline)
os.chdir('C:/Users/obriene/Projects/Text Data Analysis/PALS')
start = time.time()
#nltk.download('all')

################################################################################
                                ##Read in Data##
################################################################################

    ####Complaints
complaints = pd.read_excel('Formal Complaints Mar 24-25.xlsx')
#Sort out issues due to merged descriptions covering multilpe lines.  Forward
#fill id and date to work out which rows correspond to which complaint
#(so blanks aren't written over), then forward fill values within each id and
#Put one row per complaint, and group up location, serviceline, subect etc
# columns into a list
complaints[['ID', 'First received']] = complaints[['ID', 'First received']
                                                  ].ffill()
complaints = (complaints.set_index(['ID', 'First received'])
              .groupby(['ID', 'First received']).ffill().reset_index()
              .groupby(['ID', 'First received', 'Description'], as_index=False)
              [['Service Line', 'Location', 'Specialty admitted',
                'Location (type)', 'Subjects', 'Sub-subject']]
              .agg(lambda x:[i for i in x if i==i])
              ).dropna(subset='Description')
#Remove PREVIOUS PALS from text where possible
complaints['Description'] = complaints['Description'].str.replace(
                            r'^.{,15}\w+\sPALS\s(.*)\n', '', regex=True)

    ####Correspondence
corresponence = pd.read_excel('PALS correspondence Mar 24-25.xlsx'
                              ).dropna(subset='Description')

################################################################################
                                     ##Set up lists##
################################################################################
word_cloud_stopwords = STOPWORDS.union(
                       {"will", "may", "dont", "whilst", "one", "now", "said",
                        "let", "go", "pt", "pts", "pals", "ed", "concerns",
                        "concern", "concerned", "raise", "raised", "raising",
                        "regarding", "wishes", "without", "feel", "feels",
                        "felt", "made", "ask", "asked", "asking", "though",
                        "got", "took", "house", "going", "still", "intends",
                        "onto", "within", "attended", "mum", "mother",
                        "mothers", "dad", "father", "fathers", "son",
                        "daughter", "child", "sibling", "siblings", "wife",
                        "husband", "family", "patient", "patients",
                        "complainant", "hospital"})

remove_aspects = (['disease', 'patient', 'patients', 'pt', 'pts', 'wish',
                   'concern', 'concerned', 'find', 'advise', 'carer',
                   'complainant', 'way', 'mention', 'consideration', 'raise',
                   'feel', 'question', 'answer', 'date', 'hour', 'day', 'week',
                   'month', 'year', 'daily', 'recently', 'regular', 'basis',
                   'monthly', 'constant', 'follow', 'complaint',
                   'inconvenience', 'need', 'problem', 'he', 'future', 'cause',
                   'result', 'comment', 'previous', 'mp', 'state', 'fact',
                   'effect', 'link', 'pal', 'pals', 'point', 'view', 'thing',
                   'state', 'explanation', 'officially', 'state', 'show',
                   'instate', 'tell', 'mention', 'look', 'ask', 'behalf',
                   'hand', 'regard', 'enquiry', 'stage', 'th', 'nothing', 'lot',
                   'lack', 'one', 'two', 'several', 'anything', 'action',
                   'status', 'i', 'I', 'son', 'daughter', 'child', 'mum',
                   'mother', 'dad', 'father', 'wife', 'husband', 'husbands',
                   'partner', 'family', 'sister', 'brother', 'sibling',
                   'grandfather', 'grandmother', 'grandson', 'granddaughter',
                   'soninlaw', 'person', 'member', 'public', 'gent',
                   'gentleman', 'lady', 'kin', 'relative', 'paul', 'friend',
                   'someone', 'nhs', 'website'] 
                  + [month.lower() for month in calendar.month_name]
                  + [day.lower() for day in calendar.day_name])

#sentiment analysis pipeline
sentiment_pipeline = pipeline(model="juliensimon/reviews-sentiment-analysis")
#nlp pipeline (for getting aspects)
nlp = spacy.load('en_core_web_trf')
#aspect analysis pipeline
model_name = "yangheng/deberta-v3-base-absa-v1.1"#"yangheng/deberta-v3-large-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

################################################################################
                                 ##Set up functions##
################################################################################

def replace_digits_except_111(text):
    # Use re.sub to replace digits with an empty string, but preserve '111'
    def replace(match):
        # If the matched number is '111', don't replace it
        if match.group(0) == '111':
            return match.group(0)
        return ""  # Replace other numbers with an empty string
    return re.sub(r'\d+', replace, text)


def preprocess_text(text):
    #remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text.lower())
    #remove numbers (except)
    text = replace_digits_except_111(text)
    #lemmatise
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    #join tokens back into a string
    processed_text = ' '.join(tokens)
    return processed_text

#Sentiment analysis using text blob
def categorise_sentiment(lst):
    categories = np.select([np.array(lst) < 0, np.array(lst) == 0,
                            np.array(lst) > 0],
                            ['Negative', 'Neutral', 'Positive'], '')
    return categories

def create_wordcloud(text_responses, stopwords, name, path):
    full_text = ' \n '.join(text_responses)
    #split into tokens, transform and join everything back into one string
    #for the word cloud
    tokens = full_text.split()
    tokens = [token.translate(str.maketrans('', '', string.punctuation)).lower()
              for token in tokens]
    joined_string = ' '.join(tokens)
    #create and save wordcloud
    wordcloud = WordCloud(width=1800, height=1800, background_color='white',
                        stopwords=stopwords, min_font_size=40, colormap='winter'
                        ).generate(joined_string)

    plt.figure(figsize=(30,40))
    plt.title(name, {'fontsize':55, 'fontweight':'bold'})
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.savefig(f'{path}/{name} wordcloud.png')
    plt.close()

def aspects_plot(df, name, area, output_path):
    #Explode lists of aspects and their sentiments, get the counts of how many
    #times these appear
    asp_sens_df = (df.explode(['Aspects', 'Aspect Sentiments'])
                   .groupby(['Aspects', 'Aspect Sentiments'], as_index=False)
                   ['ID'].count()
                   .pivot(columns='Aspect Sentiments', index='Aspects',
                   values='ID'))
    asp_sens_df['total'] = asp_sens_df.sum(axis=1)
    #Pick out top n aspects to plot
    asp_sens_df = asp_sens_df.sort_values(by='total', ascending=False
                                          ).head(25).drop('total', axis=1)
    #Create plot - if one sentiment is missing, ensure the colours still match
    colour_dict = {'Negative':'#BB2C2C', 'Neutral':'#FFCC66',
                   'Positive':'#479D4B'}
    colours = [colour_dict[col] for col in asp_sens_df.columns]
    fig, ax = plt.subplots(figsize=(15,20))
    asp_sens_df.plot(kind='barh', color=colours, ax=ax)
    plt.tick_params(axis='both',  which='major', labelsize=24)
    plt.xlabel('Count of Appearances', fontsize=24)
    plt.ylabel('Aspect', fontsize=24)
    plt.title(f'{name} - {area} Aspect Sentiment Analysis', fontsize=24)
    plt.legend(prop={'size':24})
    plt.savefig(f'{output_path}/{area} Aspect Sentiment Analysis.png',
                bbox_inches='tight', dpi=1200)
    plt.close()

def check_and_make_dir(dir_path):
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

################################################################################
                            ##For loop for analysis##
################################################################################
datasets = [(complaints, 'Formal Complaints',
             ['Service Line', 'Specialty admitted', 'Subjects']),
            (corresponence, 'PALS correspondance',
             ['Care Group', 'Subject (primary)', 'Type'])]
dataframes = []
for data in datasets:
    t0 = time.time()
    df, name, groupings = data
    output_path = f'outputs/{name}'
    check_and_make_dir(output_path)
    print(f'Running {name}')
    ################################Word Cloud##################################
    text_responses = df['Description'].str.strip()
    create_wordcloud(text_responses, word_cloud_stopwords, name, output_path)

    ##############################Sentiment Analysis############################
    #results dataframe and process text, and results dictionaries.
    df['Processed Text'] = text_responses.apply(preprocess_text)
    texts = df['Processed Text'].tolist()
 
    #Overall sentiment analysis
    hugging_dict = sentiment_pipeline(texts)
    hugging_score = [i['score'] for i in hugging_dict]
    hugging_cat = ['Positive' if i['label'] == 'LABEL_1'
                   else 'Negative' for i in hugging_dict]
    df['Overall Sentiment'] = hugging_cat

    ##########################Aspect Sentiment Analysis#########################
    #empty dicts for results
    aspects_dict = {k:[] for k in df.index}
    sentiments_dict = {k:[] for k in df.index}
    scores_dict = {k:[] for k in df.index}
 
    #Get aspects
    start_nouns = []
    aspects = []
    for text in texts:
        asps = []
        #If there is 'lack of x' in the text, ensure this is captured as one string.
        if 'lack of' in text:
            for lack_str in text.split('lack of')[1:]:
                asps.append('lack of ' + lack_str.strip().split(' ')[0])

        #Get the nouns/noun chunks to pull aspects from the text.
        clean_doc = nlp(text)
        txt_nouns = [chunk.text for chunk in clean_doc.noun_chunks]
        start_nouns += [txt_nouns]

        #remove stopwords and remove_aspects, re-chunk long nouns
        for noun_chunk in txt_nouns:
            chunk_len = len(noun_chunk.split(' '))
            noun_chunk = ' '.join([word for word in noun_chunk.split()
                                if (word not in STOPWORDS)
                                and (word not in remove_aspects)])
            if noun_chunk:
                #if a long chunk, try to re-chunk
                if chunk_len > 4:
                    asps += [chunk.text for chunk
                             in nlp(noun_chunk).noun_chunks]
                else:
                    asps.append(noun_chunk)
        aspects.append(list(set(asps)))
    #add to dataframe
    df['Inital Aspects'] = start_nouns
    df['Filtered Aspects'] = aspects

    #work out which aspects only appear once in the entire data, so we can
    #reduce the run time by not asking the sentiment of these.
    aspect_counts = pd.DataFrame([x for xs in aspects
                                  for x in xs]).value_counts()
    common_aspects = [asp[0] for asp
                      in aspect_counts.loc[aspect_counts > 1].index.values]
    #loop through each common aspect, find the corresponding texts and calculate
    #their sentiments.
    for aspect in common_aspects:
        #Get the texts that contain those aspects, calculate their sentiments
        asp_texts = df.loc[df['Filtered Aspects'].apply(lambda x: aspect in x),
                           'Processed Text']
        sentiments = classifier(asp_texts.to_list(), text_pair=aspect)
        #record the results in the dictionaries
        for i, text_col in enumerate(list(asp_texts.items())):
            txt_idx = text_col[0]
            aspects_dict[txt_idx].append(aspect)
            sentiments_dict[txt_idx].append(sentiments[i]['label'])
            scores_dict[txt_idx].append(sentiments[i]['score'])

    #Join results onto dataframe
    df = (df.join(pd.DataFrame([str(i) for i in aspects_dict.values()],
            index=aspects_dict.keys(), columns=['Aspects']))
            .join(pd.DataFrame([str(i) for i in sentiments_dict.values()],
            index=sentiments_dict.keys(), columns=['Aspect Sentiments']))
            .join(pd.DataFrame([str(i) for i in scores_dict.values()],
            index=scores_dict.keys(), columns=['Scores'])))
    df['Aspects'] = df['Aspects'].apply(ast.literal_eval)
    df['Aspect Sentiments'] = df['Aspect Sentiments'].apply(ast.literal_eval)
    df['Scores'] = df['Scores'].apply(ast.literal_eval)

    #Plot aspect bar chart
    aspects_plot(df, name, 'All', output_path)

    #Append results to list of dataframes
    dataframes.append(df)

    #######################Word clouds and plots by area########################
    for col in groupings:
        #Create directory for the column's outputs
        group_path = f'outputs/{name}/{col}'
        check_and_make_dir(group_path)

        #Get all the different categories to loop over
        cats = df[col].dropna().drop_duplicates().to_list()
        lst=False
        if isinstance(cats[0], list):
            cats = list(set([x for xs in cats for x in xs if x==x]))
            lst=True

        #Create a word cloud and aspect plot for each catogory if responses allow.
        for cat in cats:
            if lst:
                filtered_df = df.loc[df[col].apply(lambda x: cat in x)].copy()
            else:
                filtered_df = df.loc[df[col]== cat].copy()
            if len(filtered_df) >= 10:
                create_wordcloud(filtered_df['Description'].copy().str.strip(),
                                 word_cloud_stopwords.union(
                                     set(cat.split(' '))),
                                         cat.replace('/', 'or'), group_path)
                aspects_plot(filtered_df, name, cat.replace('/', 'or'),
                             group_path)
    t1 = time.time()
    print(f'{name} complete in {(t1-t0)/60:.2f} mins')


#Write to excel
writer = pd.ExcelWriter('Outputs/Sentiment.xlsx', engine='xlsxwriter')   
dataframes[0].to_excel(writer, sheet_name='Complaints', index=False)
dataframes[1].to_excel(writer, sheet_name='Corresondance', index=False)
writer.close()
end = time.time()
print(f'Complete in {(end-start)/60:.2f} mins')