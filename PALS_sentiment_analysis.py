import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from wordcloud import WordCloud, STOPWORDS
import string
import calendar
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import time
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
os.chdir('C:/Users/obriene/Projects/PALS')
#nltk.download('all')

################################################################################
                                ##Read in Data##
################################################################################

    ####Complaints
complaints = pd.read_excel('Formal Complaints Mar 24-25.xlsx')
#Sort out issued due to merged descriptions covering multilpe lines.  Forward
#fill id and date to work out which rows correspond to which complaint
#(so blanks aren't written over), then forward fill values within each id and
#Put one row per complaint, and group up location, serviceline, subect etc
# columns into a list
complaints[['ID', 'First received']] = complaints[['ID', 'First received']].ffill()
complaints = (complaints.set_index(['ID', 'First received'])
              .groupby(['ID', 'First received']).ffill().reset_index()
              .groupby(['ID', 'First received', 'Description'], as_index=False)
              [['Service Line', 'Location', 'Specialty admitted',
                'Location (type)', 'Subjects', 'Sub-subject']]
              .agg(lambda x:[i for i in x if i==i])).dropna(subset='Description')
#Remove PREVIOUS PALS from text where possible
complaints['Description'] = complaints['Description'].str.replace(
                            r'^.{,15}\w+\sPALS\s(.*)\n', '', regex=True)

    ####Correspondence
corresponence = pd.read_excel('PALS correspondence Mar 24-25.xlsx'
                              ).dropna(subset='Description')

    ####List of Diseases
diseases = pd.read_csv('''C:/Users/obriene/.cache/kagglehub/datasets/harshdhakad20/list-of-all-the-diseases/versions/1/Diseases.csv''', delimiter=',')
diseases = [disease.split(' (')[0].lower()
            for disease in diseases.iloc[:,0].tolist()]
keep_diseases = ['treatment', 'endoscopy', 'dental', 'nursing', 'ultrasound',
                 'radiotherapy', 'rheumatology', 'hysterectomy', 'mri scan',
                 'hip replacement', 'colonoscopy', 'appendectomy', 
                 'mental health', 'chemotherapy', 'pain management',
                 'pregnancy', 'bullying', 'defibrillator', 'back surgery',
                 'breast implants', 'dialysis', 'x-rays']
diseases = [i for i in diseases if ((len(i) >= 5) and (i not in keep_diseases))]

################################################################################
                        ##Set up lists and functions##
################################################################################
word_cloud_stopwords = STOPWORDS.union(
                       {"will", "may", "dont", "whilst", "one", "now", "said",
                        "let", "go", "pt", "pts", "pals", "ed", "concerns",
                        "concern", "concerned", "raise", "raised", "raising",
                        "regarding", "wishes", "without", "feel", "feels",
                        "felt", "made", "ask", "asked", "asking", "though",
                        "got", "took", "house", "going", "still", "intends",
                        "onto", "within", "attended",
                        "mum", "mother", "mothers", "dad", "father", "fathers",
                        "son", "daughter", "child", "sibling", "siblings",
                        "wife", "husband", "family", "patient", "patients",
                        "complainant", "hospital"})

remove_aspects = (['disease', 'patient', 'pt', 'wish', 'concern', 'wa', 'lack',
                   'daughter', 'time', 'service',
                  'mother', 'team', 'husband', 'wife', 'father', 'son', 'family',
                  'year', 'month', 'day', 'week', 'feel', 'question', 'member',
                  'issue', 'home', 'relief', 'decision', 'complaint', 'eye',
                  'child', 'fact', 'list', 'explanation', 'state', 'leg',
                  'relation', 'partner', 'trust', 'failure', 'need', 'problem',
                  'granddaughter', 'comment', 'opinion', 'date', 'unit', 'area',
                  'belief', 'relate', 'level', 'action', 'access', 'post', 'risk',
                  'whilst', 'son-in-law', 'raise', 'recieve', 'ha', 'intake',
                  'consideration', 'issue', 'aware', 'one', 's', 'complainant',
                  'mention', 'request', 'doe', 'incorrect', 'line', 'towards',
                  'cause', 'someone', 'report', 'anything', 'friend', 'hold',
                  'understand', 'end', 'e', 'comlication', 'difficulty',
                  'capacity', 'relative', 'detail', 'see', 'event', 'distress',
                  'anyone', 'step', 'standard', 'instruction', 'name', 'mum',
                  'matter', 'recieve', 'arrangement', 't', 'mr', 'copy', 'take',
                  'move', 'regard', 'person', 'circumstance', 'claim', 'know',
                  'use', 'pressure', 'number', 'disorder', 'self', 'point',
                  'manner', 'responsibility', 'case', 'lot', 'baby', 'boy',
                  'nothing', 'sister', 'part', 'd', 'hand', 'something',
                  'advises', 'office', 'nt', 'stage', 'stress', 'speak',
                  'advise', 'house', 'follow', 'account', 'let', 'check',
                  'thumb', 'b', 'thing', 'brother', 'behalf', 'look', 'expense',
                  'rate', 'future', 'act', 'interest', 'query', 'side', 'order',
                  'grandmother', 'fu', 'return', 'following', 'gentleman',
                  'freedom', 'paper', 'minute', 'hour', 'lead', 'public',
                  'feeling', 'mark', 'meant', 'pre', 'm', 'relates', 'heard',
                  'hearing', 'resolution', 'respond', 'news', 'newphew', 'inn',
                  'perform', 'mp', 're', 'reaction', 'inform', 'paul',
                  'refernce', 'implication', 'impact', 'section', 'value',
                  'aunt', 'daughterinlaw', 'course', 'dad', 'summary', 'veteran',
                  'view', 'development', 'vision', 'ask', 'die', 'space',
                  'couple', 'clarity', 'c', 'till', 'grandson' ,'grandfather',
                  ''] 
                  + [month.lower() for month in calendar.month_name]
                  + [day.lower() for day in calendar.day_name])

punctuation_mapping_table = str.maketrans('', '', string.punctuation)
sentiments = []

def preprocess_text(text):
    #make lower case
    text = text.lower()
    #Replace and illness/disease mentioned to a standard 'disease' so these
    #don't get picked as aspects.
    for disease in diseases:
        if disease in text:
            text = text.replace(disease, 'disease')
    #tokenise
    tokens = word_tokenize(text)
    #lemmatise tokens (except ones where s gets removed)
    lemmatizer = WordNetLemmatizer()
    non_lematize = ['was', 'has', 'does']
    lemmatized_tokens = [lemmatizer.lemmatize(token)
                         if token not in non_lematize else token
                         for token in tokens]
    #remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    remove_punctuation = [regex.sub('', word) for word in lemmatized_tokens
                          if ((word not in string.punctuation)
                          and (word not in STOPWORDS))]
    #join tokens back into a string
    processed_text = ' '.join(remove_punctuation)
    return processed_text

#Sentiment analysis using text blob
def categorise_sentiment(lst):
    categories = np.select([np.array(lst) < 0, np.array(lst) == 0,
                            np.array(lst) > 0],
                            ['Negative', 'Neutral', 'Positive'], '')
    return categories

def create_wordcloud(text_responses, stopwords, name, path):
    full_text = ' \n '.join(text_responses)
    #split into tokens, transform and join everything back into one string for the word cloud
    tokens = full_text.split()
    tokens = [token.translate(punctuation_mapping_table).lower() for token in tokens]
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
    asp_sens_df = (df.explode(['Aspects', 'Aspect sentiment'])
                   .groupby(['Aspects', 'Aspect sentiment'], as_index=False)
                   ['ID'].count()
                   .pivot(columns='Aspect sentiment', index='Aspects',
                   values='ID'))
    asp_sens_df['total'] = asp_sens_df.sum(axis=1)
    #Pick out top n aspects to plot
    asp_sens_df = asp_sens_df.sort_values(by='total', ascending=False).head(20)
    #Create plot
    asp_sens_df[['Negative', 'Neutral', 'Positive']].plot(kind='barh', color=['#BB2C2C', '#FFCC66', '#479D4B'],
                     title=f'{name} - {area} Aspect Sentiment Analysis', figsize=(15,25))
    plt.savefig(f'{output_path}/{area} Aspect Sentiment Analysis.png', bbox_inches='tight')
    plt.close()

def check_and_make_dir(dir_path):
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

################################################################################
                            ##For loop for analysis##
################################################################################
datasets = [(complaints, 'Formal Complaints', ['Service Line', 'Specialty admitted', 'Subjects']),
            (corresponence, 'PALS correspondance', ['Care Group', 'Subject (primary)', 'Type'])]

for data in datasets:
    df, name, groupings = data
    output_path = f'outputs/{name}'
    check_and_make_dir(output_path)
    ################################Word Cloud##################################
    text_responses = df['Description'].str.strip()
    #create_wordcloud(text_responses, word_cloud_stopwords, name, output_path)

    ##############################Sentiment Analysis############################
    #results dataframe and process text
    df['Processed Text'] = text_responses.apply(preprocess_text)
 
    #Overall sentiment analysis
    sentiment_pipeline = pipeline(model="juliensimon/reviews-sentiment-analysis")
    hugging_dict = sentiment_pipeline(df['Processed Text'].tolist())
    hugging_score = [i['score'] for i in hugging_dict]
    hugging_cat = ['Positive' if i['label'] == 'LABEL_1'
                   else 'Negative' for i in hugging_dict]
    df['Overall Sentiment'] = hugging_cat

    #aspect analysis
    model_name = "yangheng/deberta-v3-large-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    all_aspects = []
    all_found_aspects = []
    for text in df['Processed Text'].tolist():
        #Extract nouns/aspects from the text
        tagged = nltk.pos_tag(text.split(' '))
        found_aspects = list(set([i[0] for i in tagged
                            if ((i[1] =='NN') or (i[1] == 'NNS'))]))
        all_aspects += found_aspects
        all_found_aspects.append(found_aspects)
    #record the initial list of aspects from just pickings nouns from tagging
    df['Initial Aspects'] = all_found_aspects
    #work out which aspects only appear once in the entire data, so we can
    #reduce the run time by not asking the sentiment of these.
    aspect_counts = pd.DataFrame(all_aspects).value_counts()
    uncommon_aspects = [asp[0] for asp
                        in aspect_counts.loc[aspect_counts == 1].index.values]

    all_aspects = []
    asp_sentiments = []
    asp_scores = []
    for text, aspects in df[['Processed Text', 'Initial Aspects']].values:
        #Go through each aspect and see whay the sentiment of it is.
        #remove aspects that are uncommon or we don't want to look at
        aspects = [asp for asp in aspects
                   if ((asp not in uncommon_aspects)
                   and (asp not in remove_aspects))]
        #THIS IS VERY SLOW, HOW CAN WE IMPROVE?
        aspect_sents = []
        aspect_scores = []
        for aspect in set(aspects):
            sentiment = classifier(text,  text_pair=aspect)[0]
            aspect_sents.append(sentiment['label'])
            aspect_scores.append(sentiment['score'])
        all_aspects.append(aspects)
        asp_sentiments.append(aspect_sents)
        asp_scores.append(aspect_scores)
    df['Aspects'] = all_aspects
    df['Aspect sentiment'] = asp_sentiments
    df['Aspect score'] = asp_scores

    #Plot aspect bar chart
    aspects_plot(df, name, 'All', output_path)

    #Append results to list of dataframes
    sentiments.append(df)

    ############################Word clouds by area#############################
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
                text_responses = df.loc[df[col].apply(lambda x: cat in x),
                                        'Description'].copy().str.strip()
            else:
                text_responses = df.loc[df[col]== cat, 'Description'
                                        ].copy().str.strip()
            if len(text_responses) >= 10:
                create_wordcloud(text_responses,
                                 word_cloud_stopwords.union(set(cat.split(' '))),
                                 cat.replace('/', 'or'), group_path)
                aspects_plot(text_responses, name, cat, output_path)

        






#Write to excel
start_row = 0
writer = pd.ExcelWriter('Outputs/Sentiment.xlsx', engine='xlsxwriter')   
workbook=writer.book
worksheet=workbook.add_worksheet('Full Data')
writer.sheets['Full Data'] = worksheet
for df in sentiments:
    df.to_excel(writer, sheet_name='Full Data', startrow=start_row, startcol=0,
                index=False)
    start_row += (len(df) + 2)
writer.close()
