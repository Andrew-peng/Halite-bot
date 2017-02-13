import json
import sys
import requests
import os

playerID = sys.argv[1]
limit = sys.argv[2]
outputfolder = sys.argv[3]

jsonurl = "https://halite.io/api/web/game?userID=" + playerID + "&limit=" + str(limit)

resp = requests.get(jsonurl)
if resp.status_code == 200 and len(resp.json()) > 0:
    data = json.loads(resp.text)
    for elem in data:
        replayName = elem["replayName"]
        fname = "./" + outputfolder + "/" + replayName
        if replayName is None or os.path.isfile(fname):
            continue
        for user in elem["users"]:
            if user["userID"] == str(playerID) and user["rank"] == str(1):
                print("Downloading: " + str(replayName))
                tempurl = "https://s3.amazonaws.com/halitereplaybucket/" + str(replayName)
                req = requests.get(tempurl)
                f = open(fname, "wb")
                for chunk in req.iter_content(chunk_size=1000):
                    f.write(chunk)
                f.close()
