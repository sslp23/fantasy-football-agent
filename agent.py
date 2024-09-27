import instructor
import google.generativeai as genai
from pydantic import BaseModel, Field, create_model
from typing import Callable, Literal

from utils import *
from scraper import *
from sleeper import *
import pandas as pd
import inspect
import ast
import time

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

class ClassificationResponse(BaseModel):
    """
    A few-shot example of text classification:

    Example:
    - "You should analyse in perspective of: Jordan Mason \n49ers head coach Kyle Shanahan said Christian McCaffrey (calf/Achilles) will not play in Week 2 and the team is considering placing him on injured reserve.\n             Per Shanahan, McCaffrey had his worst day in terms of pain and health following Thursday’s practice, prompting the change in tone from the team. Shanahan had previously told reporters that McCaffrey would not go on injured reserve. The superstar running back was even practicing in a limited capacity earlier this week. It’s a nightmare scenario for fantasy managers who were expecting to get McCaffrey back sooner rather than later. He will miss at least four games if placed on injured reserve. Jordan Mason will dominate the touches for San Francisco for the foreseeable future. Mason turned 28 carries into 147 yards and a touchdown in Week 1. He will rank as a high-end RB1 for as long as McCaffrey is on the sidelines.": POSITIVE
    - "You should analyse in perspective of: Christian McCaffrey \n49ers head coach Kyle Shanahan said Christian McCaffrey (calf/Achilles) will not play in Week 2 and the team is considering placing him on injured reserve.\n             Per Shanahan, McCaffrey had his worst day in terms of pain and health following Thursday’s practice, prompting the change in tone from the team. Shanahan had previously told reporters that McCaffrey would not go on injured reserve. The superstar running back was even practicing in a limited capacity earlier this week. It’s a nightmare scenario for fantasy managers who were expecting to get McCaffrey back sooner rather than later. He will miss at least four games if placed on injured reserve. Jordan Mason will dominate the touches for San Francisco for the foreseeable future. Mason turned 28 carries into 147 yards and a touchdown in Week 1. He will rank as a high-end RB1 for as long as McCaffrey is on the sidelines.": NEGATIVE
    - "You should analyse in perspective of: Travis Etienne Jr. \nTravis Etienne Jr. rushed 11 times for 68 yards in a Week 3 loss to the Bills, adding 17 receiving yards on four catches. \n             On a positive note, Etienne had 11 carries and Tank Bigsby had just two, so there was no sense of a committee happening here. On the other hand, the Jaguars were down big in a heartbeat and all but abandoned the running game before a few big garbage-time runs by Etienne against the Bills’ backup defenders. Etienne also had six targets in the passing game, so his role in this offense is secure, but the struggles on the offensive line create some cause for concern in Etienne’s season-long outlook. He’s now on the RB1/RB2 border.": NEUTRAL
    - "You should analyse in perspective of: Tank Bigsby \nTravis Etienne Jr. rushed 11 times for 68 yards in a Week 3 loss to the Bills, adding 17 receiving yards on four catches. \n             On a positive note, Etienne had 11 carries and Tank Bigsby had just two, so there was no sense of a committee happening here. On the other hand, the Jaguars were down big in a heartbeat and all but abandoned the running game before a few big garbage-time runs by Etienne against the Bills’ backup defenders. Etienne also had six targets in the passing game, so his role in this offense is secure, but the struggles on the offensive line create some cause for concern in Etienne’s season-long outlook. He’s now on the RB1/RB2 border.": NEGATIVE
    - "You should analyse in perspective of: Christian McCaffrey \n49ers HC Kyle Shanahan said RB Christian McCaffrey (Achilles) will see a specialist in Germany for his Achilles.\n             McCaffrey, who was placed on injured reserve earlier in the season, was already reportedly looking at a six week or more absence due to Achilles tendinitis, with the reports now coming in that he will see a specialist in Germany to further address the issue. He initially attempted to play through the ailment but was placed on injured reserve following a flare up after practice before the team’s Week 2 contest. This certainly does not sound promising for the reigning touchdown leader moving forward.": NEGATIVE
    - "You should analyse in perspective of: Kenneth Walker III \nZach Charbonnet had 18 rushes for 91 yards and two touchdowns in the Seahawks’ Week 3 win against the Dolphins, adding three catches for 16 yards. \n             Though Charbonnet wasn’t particularly impressive, he once again benefited from touch volume with Ken Walker (oblique) on the sideline. Charbonnet ran tough, as usual, and cashed in two inside-the-10 rushes for a pair of scores against an exploitable Miami run defense. He had 18 of the team’s 21 running back rushes. As long as Walker is out, Charbonnet profiles as a top-12 fantasy play with plenty of touchdown upside in a potent Seahawks offense. ": NEGATIVE
    - "You should analyse in perspective of: Zach Charbonnet \nZach Charbonnet had 18 rushes for 91 yards and two touchdowns in the Seahawks’ Week 3 win against the Dolphins, adding three catches for 16 yards. \n             Though Charbonnet wasn’t particularly impressive, he once again benefited from touch volume with Ken Walker (oblique) on the sideline. Charbonnet ran tough, as usual, and cashed in two inside-the-10 rushes for a pair of scores against an exploitable Miami run defense. He had 18 of the team’s 21 running back rushes. As long as Walker is out, Charbonnet profiles as a top-12 fantasy play with plenty of touchdown upside in a potent Seahawks offense. ": Positive
    """
    chain_of_thought: str = Field(
        ...,
        description="Think step by step to determine the correct label.",
    )
    
    label: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"] = Field(
        ...,
        description="The predicted class label.",
    )

class BasicOutput(BaseModel):
    output: str

def get_news(player_name: str, link: str):
    link = link+'/news'
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')

    headlines = soup.find('ul', {'class':'PlayerNewsModuleList-items'}).find_all('div', {'class': 'PlayerNewsPost-headline'})[:2]
    texts = soup.find('ul', {'class':'PlayerNewsModuleList-items'}).find_all('div', {'class': 'PlayerNewsPost-analysis'})[:2]

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

    def basic_qa(self, prompt: str, Model):
        return self.ai.chat.completions.create(
                response_model=Model,
                messages=[
                    {"role": "user","content": prompt},
                ],
            )
    
    def retrieve(self, prompt: str, context: str, Model):
        return self.ai.chat.completions.create(
                response_model=Model,
                messages=[
                    {"role": "system","content": "Based on the context provided, you need to retrieve the informations of the player."},
                    {"role": "user", "content": context},
                    {"role": "user","content": prompt},
                ],
            )
    
    def planner(self, prompt: str, Model) -> Plan:
        return self.ai.chat.completions.create(
                response_model=Model,
                messages=[
                    {"role": "system","content": "Based on the context provided, you need to plan what agents will be used."},
                    {"role": "user","content": prompt},
                ],
            )
    
    def news_classify(self, player_news: list) -> list:
        """Perform single-label classification on the input text."""
        
        news_with_labels = []
        counter = 0
        for news in player_news:
            labels = []
            reasons = []
            for (n, h) in zip(news['News'], news['Headlines']):
                int_player = news['Player']
                print(f'Classifying news from: {int_player}')
                if counter>=5:
                    print('Max rate - sleeping 1min')
                    time.sleep(60)
                    counter=0
                label = self.ai.chat.completions.create(
                    response_model=ClassificationResponse,
                    messages=[
                        {"role": "user","content": f"Classify the following text: <text>You should analyse in perspective of: {int_player} \n{n}</text>",},
                    ],
                )
                labels.append(label.label)
                reasons.append(label.chain_of_thought)
                counter+=1
            news['Label'] = labels
            news['Reasoning'] = reasons
            news_with_labels.append(news)
        
        return news_with_labels
    
    def news_agent(self, players: FantasyInfos) -> list:
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

    def get_function_details(self, func):
        # Get the signature of the function
        signature = inspect.signature(func)
        
        # Create a list of tuples (argument, argument_type)
        arg_details = [
            (param_name, str(param.annotation.__name__) if param.annotation != inspect.Parameter.empty else 'Any')
            for param_name, param in signature.parameters.items()
        ]
        
        # Convert the list of tuples to a string with comma separation
        #arg_details_str = ', '.join([f"{arg}, type: {arg_type}" for arg, arg_type in arg_details])
        #arg_details_args = ', '.join([f"({arg_type})" for arg, arg_type in arg_details])
        #print(arg_details)
        arg_details_str = pd.DataFrame([(arg,arg_type) for arg, arg_type in arg_details], columns=['Name', 'Type']).to_markdown()
        return arg_details_str

    def run(self, prompt):
        #all_rosters_data = self.rosters[['PLAYER_NAME', 'POSITION', 'OWNER_NAME']]
        #my_roster_data = self.user_roster[['PLAYER_NAME', 'POSITION', 'OWNER_NAME']]

        plan_instruction = f'''
        Your goal is to select which agents will be used to answer the question: {prompt}

        You have access to the following agents:

        - news_agent -> retrieve news of the players
        - news_classify -> label the news as positive or negative

        If the answer can be answered without this agent, you don't need to use it, so return an empty string.
        You don't need to use all agents. Use only the one explicit found in the question.
        
        An example of output:
        agent1->agent2->agent3
        '''
        
        player_retrieve_context = f"# User roster: {self.user_roster.to_markdown()}\n # Other rosters: {self.rosters.to_markdown()}"
        
        players = self.retrieve(prompt, player_retrieve_context, FantasyInfos)
        plan = self.planner(plan_instruction, Plan)

        agents = plan.agents
        outputs = []
        for agent in agents.split('->'):
            func_args = self.get_function_details(getattr(self, agent))
            local_vars = locals()
            #possible_params = ', '.join([f"{var}, type: {type(value).__name__}" for var, value in local_vars.items()])
            possible_params = pd.DataFrame([(var, type(value).__name__) for var, value in local_vars.items()], columns=['Name', 'Type']).to_markdown()
            exp = f'''
                You must find the parameters for function {agent} based on the question {prompt} and the plan {agents}.
                The function agent expect the following pattern of parameters:
                {func_args}

                You must select based only on the values below. The parameters selected must have the same type, but can have a different name:
                {possible_params}
                
                
                You must return an str with arguments separated by ",".
            '''
            args = self.basic_qa(exp, BasicOutput).output

            print(agent, args)
            args_vars = []
            for arg in args.split(','):
                args_vars.append(locals()[arg])
            
            output = getattr(self, agent)(*tuple(args_vars))
            outputs.append(output)
            locals()[agent+'_output'] = output

        return outputs
    
a = Agent(LEAGUE_ID, USER_ID)

test = a.run('Get the news of my team labeled as positive or negative.')

test

a.news_agent('Jordan Mason')
