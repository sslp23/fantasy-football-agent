import pandas as pd
from sleeper_wrapper import League, Players

def parse_rosters(l, pls, owners):
    rosters = l.get_rosters()
    teams = pd.DataFrame()
    for team in rosters:
        players = team['players']
        owner = [team['owner_id']]*len(players)

        df_team = pd.DataFrame({'PLAYER_ID': players, 'OWNER_ID':owner})

        teams = pd.concat([teams, df_team])

    teams = teams.merge(pls, how='left', on='PLAYER_ID')
    teams = teams.merge(owners, how='left', on='OWNER_ID')

    return teams

def league_infos(league_id):
    league = League(league_id)
    
    teams = league.get_users()
    owners_df = pd.DataFrame([(a['display_name'], a['user_id']) for a in teams], columns=['OWNER_NAME', 'OWNER_ID'])

    players = Players()
    all_players = players.get_all_players()
    ap = {}
    pid, fn, pos = [],[],[]
    for a in all_players.keys():
        if all_players[a]['sport'] == 'nfl':
            pid.append(a)
            fn.append(all_players[a]['first_name']+' '+all_players[a]['last_name'])
            pos.append(all_players[a]['position'])

        
    ap['PLAYER_ID'] = pid
    ap['PLAYER_NAME'] = fn
    ap['POSITION'] = pos

    ap_df = pd.DataFrame(ap)
    comp_df = parse_rosters(league, ap_df, owners_df)
    return comp_df
