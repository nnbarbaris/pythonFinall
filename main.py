import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
from IPython.display import HTML
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import networkx as nx

with st.echo(code_location='below'):
    st.title("Here are some statistics about Netflix movies")
    st.image("https://www.ixbt.com/img/n1/news/2022/3/3/netflix_large.jpg", width=800 )
    def print_hi(name):
        st.write(f"## Hello, {name}")
    if __name__ == "__main__":
        name = st.text_input("Введите имя ", key= "text")
        print_hi(name)
    def get_data():
        data_url = (
            "https://github.com/nnbarbaris/Netflix/raw/main/NetflixOriginals.csv"
        )
        return (
            pd.read_csv(data_url, encoding="ISO-8859-1").dropna(subset=["Premiere"]))

    table= get_data()
    table.info()
    tab=table.sort_values(by=["IMDB Score"], ascending= False).iloc[:10]
    st.write("This table shows the best 10 Netflix movies by IMDB Score")
    st.write(tab)
    #Визуализация 1
    data_names = tab['Title'].tolist()
    data_values = tab['IMDB Score'].tolist()
    dpi = 80
    mpl.rcParams.update({'font.size': 9})
    xs = range(len(data_names))
    ax = plt.axes()
    ax.xaxis.grid(True, zorder=1)
    st.write(f"## You can see here Top15 Films and their IMDB Score ")
    plt.title('Топ-10 фильмов и их рейтинг')

    bar = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
    plt.bar([x + 0.05 for x in xs], [d * 0.9 for d in data_values],
            width=0.5, color='yellow', alpha=0.9,
            zorder=5)

    plt.xlabel('Title')
    plt.ylabel('Score')
    plt.xticks(xs, data_names)
    bar.autofmt_xdate(rotation=25)
    st.pyplot(bar)
    #NUMPY
    mean = np.mean(table['IMDB Score'])
    mm = np.around(mean, decimals=3)
    st.write(f"## Average of all Netflix movies rating:", mm)

    #Сортировка по жанрам
    def which_genre():
        genres = table.groupby(['Genre']).size().reset_index(name='count')
        return (genres)
    gen=which_genre()
    gen = gen.sort_values(by=["count"], ascending=False).iloc[:10]

    # Визуализация 2
    data_nam = gen['Genre'].tolist()
    data_val = gen['count'].tolist()
    st.write(f"## Distribution of the number of films by genre")
    plt.title('Распределение фильмов по жанрам (%)')
    ### FROM: https://eax.me/python-matplotlib/
    fig = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
    plt.pie(
        data_val, autopct='%.1f', radius=1.1,
        explode=[0.15] + [0 for _ in range(len(data_nam) - 1)])
    plt.legend(
        bbox_to_anchor=(-0.16, 0.45, 0.25, 0.25),
        loc='lower left', labels=data_nam)
    ###END FROM
    st.pyplot(fig)

    #Создаю дополнительный столбец, где только год выпуска
    llist=[]
    for i in table['Premiere']:
        a=i.split(",")
        if len(a)>1:
            llist.append(a[1])
        else: llist.append(i)
    table['Year']=llist

    date = table.sort_values(by=["Year","IMDB Score" ], ascending=False).iloc[5:15]
    st.write(f"## You can see here latest news sorted by rating ")
    #Последние новинки отсортированные по рейтингу
    st.write(date[['Title', 'Year', 'IMDB Score']])
    dd=date.iloc[:3]
    proj = st.selectbox("Выберите новинку, чтобы посмотреть её обложку", dd['Title'])

    # BEAUTIFUL SOUP+ REST API
    entrypoint = "https://en.wikipedia.org/wiki/"
    if proj== "Ferry":
        st.write("Обложки данного фильма не было в том источнике, где брались первые две обложки,"
                 " поэтому если возникла ошибка, попробуйте перезагрузить или воспользуйтесь ссылкой на картинку")
        r = requests.get("https://kino.mail.ru/cinema/movies/930601_ferri/")
        soup = BeautifulSoup(r.text)
        a = [a.get('src') for a in soup.find_all("img") if a.get('src')]
        st.write(a)
        im=a[4]
        st.image(im,
                 width=200,
                 )
    else:
        if proj == "The Disciple":
            proj = 'The_Disciple_(2020_film)'
        def mkurl(name):
            return entrypoint + name
        r = requests.get(mkurl(proj))
        soup = BeautifulSoup(r.text, features="html.parser")
        a = [a.get('src') for a in soup.find_all("img") if a.get('src')]
        im = a[0]
        st.image( "https:" + im,
            width=200,
        )

    def score():
        sc = table.groupby(['IMDB Score']).size().reset_index(name='count_sc')
        return (sc)

    sc= score()
    table['count_sc']=sc['count_sc']
    sc= table.sort_values(by= ['count_sc','Year'], ascending=False).iloc[:54]

    st.write(f"## This scatterplot shows how many films gain a certain score")

    ###FROM : https://seaborn-pydata-org.translate.goog/examples/scatterplot_sizes.html?_x_tr_sl=en&_x_tr_tl=ru&_x_tr_hl=ru&_x_tr_pto=sc
    sns.set_theme(style="whitegrid")
    cmap = sns.cubehelix_palette(as_cmap=True)
    g = sns.relplot(
        data=sc,
        x="count_sc", y="IMDB Score",
        hue="count_sc", size="Year",
        palette=cmap, sizes=(100, 200),
    )
    g.set(xscale="linear", yscale="linear")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)
    ###END FROM

    plt.xticks(rotation=45)
    st.pyplot(g)
    st.write("'count_sc' shows how many movies have the same rating")
    #NETWORKX
    graph =table.iloc[:30]
    g = nx.from_pandas_edgelist(graph, source='Language', target='Title',  edge_attr ='Title')
    lang = [node for node in g.nodes() if node in graph.Language.unique()]
    lang_dict = dict(zip(lang, lang))
    st.title("Languagies and Titles")
    st.markdown("This graph shows the connection between 30 movies and their origin languages; "
                "nodes that are not signed are movies")
    fig, ax = plt.subplots()
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, node_color='yellow', with_labels=True,labels= lang_dict, font_size=14, )
    st.pyplot(fig)

    def langu():
        langu = table.groupby(['Language']).size().reset_index(name='langs')
        return (langu)
    llang=langu()
    # NUMPY
    ssum = np.sum(llang['langs'])
    ggen = llang.sort_values(by=["langs"], ascending=False).iloc[:30]
    st.write(f"## Choose a language to find out what percent of Netflix released movies match your language")
    gg = st.selectbox("", ggen['Language'])
    df_selection = llang[lambda x: x["Language"] == gg]
    s = df_selection['langs']
    ans = (s / ssum) * 100
    answ = np.around(ans, 2)
    st.write(answ)
