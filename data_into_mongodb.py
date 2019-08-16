from pymongo import MongoClient
from pprint import pprint
from os import listdir
from wd import *
import requests
from bs4 import BeautifulSoup
from copy import deepcopy
import pandas as pd

def get_episode_text(link):
    r = requests.get('https://www.springfieldspringfield.co.uk/{}'.format(link))
    soup = BeautifulSoup(r.text, 'html.parser')
    container = soup.find('div', class_='scrolling-script-container')
    return container.text.strip()


def create_episode_dicts(country):

    home = requests.get('https://www.springfieldspringfield.co.uk/episode_scripts.php?tv-show=the-office-{}'.format(country.lower()))
    hsoup = BeautifulSoup(home.text, 'html.parser')
    hcontainer = hsoup.find_all('a', class_='season-episode-title')

    links = [h.attrs['href'] for h in hcontainer]
    uk_list = []

    for link in links:
        uk_dict = {}
        uk_dict['country'] = country
        uk_dict['season'] = link[-6:-3]
        uk_dict['episode'] = link[-3:]
        uk_dict['transcript'] = get_episode_text(link)
        uk_list.append(uk_dict)

    return uk_list

uk_list = create_episode_dicts('UK')

print(len(uk_list))
uk_list[0]

us_list = create_episode_dicts('US')
print(len(us_list))
us_list[0]

office_list = deepcopy(uk_list)
type(office_list)

office_list.extend(us_list)
len(office_list)
office_list[14]

#===============================================================================================================================

client = MongoClient()

client.list_database_names()

office_db = client['office']

office_db.create_collection('all_episodes')
# office_db.drop_collection('all_episodes')
all_episodes = office_db.get_collection('all_episodes')

all_episodes.insert(office_list)

client.list_database_names()

all_episodes.count()

all_episodes.find({'episode': 'e01'}).count()

cursor = all_episodes.find({}, {'_id': 0, 'country': 0, 'season': 0, 'episode': 0}).limit(7)

for elem in cursor:
    # print(type(elem))
    print(elem)
