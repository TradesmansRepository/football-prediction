import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import time

standings_url = 'https://fbref.com/en/comps/10/Championship-Stats'
years = list(range(2023, 2020, -1))
all_matches = []

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text, features="lxml")
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"
    print(f'calculating year {year}')

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        print(f'calculating team {team_name}')
        data = requests.get(team_url)
        matches = pd.read_html(StringIO(data.text), match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text, features="lxml")
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")

        shooting = pd.read_html(StringIO(data.text), match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Championship"]
        
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(5)


print(all_matches)
len(all_matches)
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("matches3.csv")