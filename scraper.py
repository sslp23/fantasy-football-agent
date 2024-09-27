import requests
from bs4 import BeautifulSoup

def get_news(link):
    link = link+'/news'
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')

    headlines = soup.find('ul', {'class':'PlayerNewsModuleList-items'}).find_all('div', {'class': 'PlayerNewsPost-headline'})[:5]
    texts = soup.find('ul', {'class':'PlayerNewsModuleList-items'}).find_all('div', {'class': 'PlayerNewsPost-analysis'})[:5]

    news = [h.text+' '+t.text for h, t in zip(headlines, texts)]
    heads = [h.text for h in headlines]
    return news, heads

