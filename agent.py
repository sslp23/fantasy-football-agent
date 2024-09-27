import instructor
import google.generativeai as genai
from pydantic import BaseModel, Field, create_model
from typing import Callable, Literal

from utils import *
from scraper import *
from sleeper import *
import pandas as pd
import inspect

genai.configure(api_key=GEMINI_API_KEY)

class FantasyPlayers(BaseModel):
    name: str
    position: str
    owner: str

class FantasyInfos(BaseModel):
    infos: list[FantasyPlayers]    

class Plan(BaseModel):
    agents: str = Field('''
    This agent must be retrieved in the format:
                        
    agent1->agent2->agent3
    ''') #step name
    chain_of_thought: str = Field(
        ...,
        description="Think step by step to determine the correct label.",
    )

def get_news(player_name: str, link: str):
    link = link+'/news'
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')

    headlines = soup.find('ul', {'class':'PlayerNewsModuleList-items'}).find_all('div', {'class': 'PlayerNewsPost-headline'})[:5]
    texts = soup.find('ul', {'class':'PlayerNewsModuleList-items'}).find_all('div', {'class': 'PlayerNewsPost-analysis'})[:5]

    news = [h.text+' '+t.text for h, t in zip(headlines, texts)]
    heads = [h.text for h in headlines]
    news_infos = {'Player': player_name, 'News': news, 'Headlines': heads}
    return news_infos

class Agent:
    def __init__(self, league_id, team_name):
        self.ai = instructor.from_gemini(
                    client=genai.GenerativeModel(
                        model_name="models/gemini-1.5-flash-latest",
                    ),
                )
        self.league = league_id
        self.owner_name = team_name
        self.rosters = league_infos(league_id)[['PLAYER_NAME', 'POSITION', 'OWNER_NAME']]
        self.user_roster = self.rosters[self.rosters.OWNER_NAME == self.owner_name][['PLAYER_NAME', 'POSITION', 'OWNER_NAME']]

    def retrieve(self, prompt, context, Model):
        return self.ai.chat.completions.create(
                response_model=Model,
                messages=[
                    {"role": "system","content": "Based on the context provided, you need to retrieve the informations of the player."},
                    {"role": "user", "content": context},
                    {"role": "user","content": prompt},
                ],
            )
    
    def planner(self, prompt, Model):
        return self.ai.chat.completions.create(
                response_model=Model,
                messages=[
                    {"role": "system","content": "Based on the context provided, you need to plan what agents will be used."},
                    {"role": "user","content": prompt},
                ],
            )
    
    def news_agent(self, players):
        class PlayerInfo(BaseModel):
            name: str
            team: str
            link: str
            position: str

        class Squad(BaseModel):
            infos: list[PlayerInfo]

        news_data = pd.read_csv('data/cnbc_players.csv')
        all_players = []
        for player in players.infos:
            if player.position != 'DEF':
                info = self.retrieve(player.name, news_data.to_markdown(), Squad).infos[0]
                player_infos = get_news(info.name, info.link)
                
                all_players.append(player_infos)
            
        return all_players


    def run(self, prompt):
        #all_rosters_data = self.rosters[['PLAYER_NAME', 'POSITION', 'OWNER_NAME']]
        #my_roster_data = self.user_roster[['PLAYER_NAME', 'POSITION', 'OWNER_NAME']]

        plan_instruction = f'''
        Your goal is to select which agents will be used to answer the question: {prompt}

        You have access to the following agents:

        - news_agent -> retrieve news of the players

        If the answer can be answered without this agent, you don't need to use it, so return an empty string
        '''
        
        player_retrieve_context = f"# User roster: {self.user_roster.to_markdown()}\n # Other rosters: {self.rosters.to_markdown()}"
        
        players = self.retrieve(prompt, player_retrieve_context, FantasyInfos)
        plan = self.planner(plan_instruction, Plan)

        agents = plan.agents.split('->')
        for agent in agents:
            print(agent)
            output = getattr(self, agent)(players)


        return output

        return #self.news_agent(players)
    
a = Agent(LEAGUE_ID, USER_ID)

test = a.run('Get the news of my players')

a.news_agent('Jordan Mason')
