import csv

points = {}

with open('predictions.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip header row
    for row in reader:
        home_team, away_team = row[1], row[2]
        home_points, away_points = int(row[3]), int(row[4])
        # Update points for home team
        if home_team not in points:
            points[home_team] = 0
        points[home_team] += home_points
        # Update points for away team
        if away_team not in points:
            points[away_team] = 0
        points[away_team] += away_points

with open('points.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Team', 'Points'])
    for team, team_points in points.items():
        writer.writerow([team, team_points])