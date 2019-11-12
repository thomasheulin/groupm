#----------------------------------------------------------------------------------------------------------------------------

interface de visualisation avec Dash


#----------------------------------------------------------------------------------------------------------------------------


import dash
from dash.dependencies import Input, Output
from dash_table import DataTable
import dash_core_components as dcc
import dash_html_components as html
from IPython import display
import os

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
from io import BytesIO
import base64

import pandas as pd
import numpy as np

df = pd.read_csv('C:\\data\\groupm\\wc\\wordcloud\\wordcloud.csv', encoding = 'utf-8')


# A preprocess .. ##################################

df['word'] = df['word'].apply(lambda x: str(x))
df['positive'] = df['positive'].apply(lambda x: int(x))

####################################################

def fig_to_uri(in_fig, close_all=True, **save_args):
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0) 
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


app = dash.Dash()

# layout :

app.layout = html.Div([
    
    dcc.RadioItems(id='radio_media',
        options=[{'label': media, 'value': df['media'].unique().tolist().index(media)} for media in df['media'].unique().tolist()],
        value=0
    ),
    
    dcc.RadioItems(id='radio_positive',
        options=[
            {'label': 'Positif', 'value': 1},
            {'label': 'Negatif', 'value': 0}
        ],
        value=1
    ),
    
    html.Div([html.Img(id = 'cur_plot', src = '')],
             id='plot_div'),
    
    dcc.Input(id='word_imput', value='rechercher un mot...', type="text"),
    
    DataTable(
        data=[], #=map_data.to_dict('records'),
        #columns=map_data.columns,
        #row_selectable='multi',
        #selected_rows=[0],
        id='table'
    )
])

# callbacks :

@app.callback(
    [Output("table", "data"), Output("table", "columns")],
    [Input("radio_media", "value"),
     Input("radio_positive", "value"),
     Input('word_imput', 'value')]
)
def updateTable(radio_media, radio_positive, word_imput):

    # mise a jour de la table sur les 2 radio buttons + le champs de texte
    # afin d'afficher les mots dans leurs contextes
    
    update = df.loc[df['media'].isin([df['media'].unique().tolist()[radio_media]]),:]
    update = update.loc[update['positive'].isin([radio_positive]),:]
    update = update.loc[update['word'].isin([word_imput]),:]
    update = update[['word_count', 'raison']]
    return update.to_dict('records'), [{"name": i, "id": i} for i in update.columns]

@app.callback(
    Output(component_id='cur_plot', component_property='src'),
    [Input("radio_media", "value"),
     Input("radio_positive", "value")]
)
def update_graph(radio_media, radio_positive):

    # mise a jour du wordcloud sur les 2 radio buttons
    
    update = df.loc[df['media'].isin([df['media'].unique().tolist()[radio_media]]),:]
    update = update.loc[update['positive'].isin([radio_positive]),:]
    
    df_word_count = update.drop_duplicates(subset=['word'])
    df_word_count = df_word_count[['word', 'word_count']]
    
    word_freq = pd.Series(df_word_count.word_count.values,index=df_word_count.word).to_dict()
    
    wordcloud = WordCloud(background_color='white', normalize_plurals=False).generate_from_frequencies(word_freq)
    
    fig, axes = plt.subplots()
    axes.imshow(wordcloud, interpolation='bilinear')
    axes.axis('off')
    out_url = fig_to_uri(fig)
    return out_url

if __name__ == '__main__':
    app.run_server(debug=True)
